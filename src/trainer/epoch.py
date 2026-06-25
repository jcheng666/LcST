"""Epoch execution: training / validation / testing loops, aux-pool evaluation."""

import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from utils.metrics import cal_metrics, target_diag_stats
from normalized_batch import NormalizedBatch
from window import (
    compute_loss,
    gather_target_instances,
    target_instance_chunks,
)

# ---------------------------------------------------------------------------
# Aux-pool evaluation
# ---------------------------------------------------------------------------


def routine_aux_pool_sets(all_aux_pool_sets, args):
    return all_aux_pool_sets[: args.routine_aux_pool_sets]


def eval_aux(model, aux_pool_sets, step_fn, agg):
    """Run step_fn under each aux_pool config, aggregate results with agg."""
    items = []
    for aux_pools in aux_pool_sets:
        model.set_aux_pools(aux_pools)
        items.append(step_fn())
    return agg(items)


# ---------------------------------------------------------------------------
# Eval-budget helpers
# ---------------------------------------------------------------------------


def _ceil_div(a, b):
    return (a + b - 1) // b


def train_steps_per_epoch(args, data_bundles):
    if args.steps_per_epoch > 0:
        return args.steps_per_epoch
    steps_per_bundle = _ceil_div(args.train_target_instances, args.batch_size)
    return steps_per_bundle * len(data_bundles)


def eval_chunk_size(args):
    return args.backbone_capacity


def _budget_window_count(loader, eval_instance_budget):
    dataset = getattr(loader, "dataset", None)
    if eval_instance_budget is None or dataset is None:
        return None
    dataset_len = len(dataset)
    if dataset_len == 0:
        return 0
    first_item = dataset[0]
    node_count = first_item[0].shape[1]
    return min(dataset_len, math.ceil(eval_instance_budget / node_count))


def _uniform_sample_indices(dataset_len, window_count):
    if window_count >= dataset_len:
        return list(range(dataset_len))
    return torch.linspace(0, dataset_len - 1, steps=window_count).round().long().tolist()


def _budget_subsample_loader(loader, eval_instance_budget):
    dataset = getattr(loader, "dataset", None)
    window_count = _budget_window_count(loader, eval_instance_budget)
    if eval_instance_budget is None or dataset is None or window_count is None:
        return loader
    dataset_len = len(dataset)
    if window_count >= dataset_len:
        return loader
    kwargs = {
        "batch_size": loader.batch_size,
        "shuffle": False,
        "drop_last": False,
        "collate_fn": loader.collate_fn,
        "num_workers": loader.num_workers,
        "pin_memory": loader.pin_memory,
    }
    if loader.num_workers > 0:
        kwargs["prefetch_factor"] = loader.prefetch_factor
    indices = _uniform_sample_indices(dataset_len, window_count)
    return DataLoader(Subset(dataset, indices), **kwargs)


# ---------------------------------------------------------------------------
# Window preparation
# ---------------------------------------------------------------------------


def unpack_batch(batch):
    if len(batch) == 2:
        input, target = batch
        return input, target, None, None
    if len(batch) == 3:
        input, target, mask = batch
        return input, target, mask, None
    if len(batch) == 4:
        return batch
    raise ValueError(f"batch must contain 2, 3, or 4 tensors, got {len(batch)}")


# ---------------------------------------------------------------------------
# Epoch loops
# ---------------------------------------------------------------------------


def TrainEpoch(
    loader,
    model,
    optim,
    loss_fn,
    args,
    is_training: bool,
    resample=True,
    forward_budget=None,
    eval_instance_budget=None,
):
    device = next(model.parameters()).device
    if is_training:
        model.train()
    else:
        model.eval()
    if resample:
        model.resample_aux_pools()

    loss_total = 0.0
    element_total = 0
    chunk_count = 0
    target_batch_size = args.batch_size if is_training else eval_chunk_size(args)

    batch_iter = loader
    if not is_training:
        batch_iter = _budget_subsample_loader(loader, eval_instance_budget)

    for batch in batch_iter:
        input, target, context_mask, eval_mask = unpack_batch(batch)
        nb = NormalizedBatch(input, target, context_mask, args, device, eval_mask=eval_mask)
        model_input = nb.model_input
        B, N, _ = model_input.shape
        # Full encode: encode all nodes once, shared across all chunks
        with (
            torch.set_grad_enabled(is_training),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True),
        ):
            bank = model.encode(model_input, mask=nb.tokenizer_mask)

        eval_full_nodes = not is_training
        for sample_ids, node_ids in target_instance_chunks(B, N, target_batch_size, device, full=eval_full_nodes):
            if forward_budget is not None and chunk_count >= forward_budget:
                break
            target_sample = gather_target_instances(nb.target, sample_ids, node_ids)

            # For impute_full: gather loss mask so we only compute loss on
            # truly-missing positions (mask==0), not on observed positions.
            loss_mask_sample = None
            if nb.mask is not None:
                loss_mask_sample = gather_target_instances(nb.mask, sample_ids, node_ids)

            possible_count = target_sample.numel()
            if loss_mask_sample is not None:
                possible_count = (loss_mask_sample == 0).sum().item()
                if possible_count == 0:
                    continue

            with (
                torch.set_grad_enabled(is_training),
                torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True),
            ):
                predict = model.predict(bank, sample_ids, node_ids)
                predict = predict.view(sample_ids.numel(), -1, args.output_dim)
                predict = nb.restore(predict, sample_ids, node_ids)
                loss_mean = compute_loss(predict, target_sample, loss_fn, mask=loss_mask_sample)
            loss_total += loss_mean.detach().item() * possible_count
            element_total += possible_count

            if is_training:
                loss_mean.backward()
                optim.step()
                optim.zero_grad()

            chunk_count += 1

        if forward_budget is not None and chunk_count >= forward_budget:
            break

    if not element_total:
        return None
    return loss_total / element_total


def train_one_step(data_bundles, train_states, model, optim, loss_fn, args, rng):
    exposure_weights = [len(bundle["train_loader"]) * bundle["node_num"] for bundle in data_bundles]
    state_ids = list(range(len(data_bundles)))
    state_idx = rng.choices(state_ids, weights=exposure_weights, k=1)[0]
    bundle = data_bundles[state_idx]
    state = train_states[state_idx]

    try:
        batch = next(state["loader_iter"])
    except StopIteration:
        state["loader_iter"] = iter(bundle["train_loader"])
        batch = next(state["loader_iter"])

    model.set_graph(bundle["key"])
    train_loss = TrainEpoch(
        [batch], model, optim, loss_fn, args,
        is_training=True, resample=False, forward_budget=1,
    )
    return bundle["key"], train_loss


def TestEpoch(
    loader, model, args, log_dir=None, save=False, resample=True, eval_instance_budget=None, full_nodes=False
):
    device = next(model.parameters()).device

    with torch.no_grad():
        model.eval()
        if resample:
            model.resample_aux_pools()
        targets = []
        predicts = []
        eval_masks = []

        target_batch_size = eval_chunk_size(args)
        batch_iter = _budget_subsample_loader(loader, eval_instance_budget)
        eval_full_nodes = full_nodes or eval_instance_budget is not None
        for batch in batch_iter:
            input, target, context_mask, eval_mask = unpack_batch(batch)
            nb = NormalizedBatch(input, target, context_mask, args, device, eval_mask=eval_mask)
            model_input = nb.model_input
            B, N, _ = model_input.shape

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                bank = model.encode(model_input, mask=nb.tokenizer_mask)  # target_node_ids=None → full encode
                for sample_ids, node_ids in target_instance_chunks(
                    B, N, target_batch_size, device, full=eval_full_nodes
                ):
                    predict = model.predict(bank, sample_ids, node_ids)
                    predict = predict.view(sample_ids.numel(), -1, args.output_dim)
                    predict = nb.restore(predict, sample_ids, node_ids)
                    target_sample = gather_target_instances(nb.target, sample_ids, node_ids)
                    targets.append(target_sample.detach())
                    predicts.append(predict.detach())
                    if nb.mask is not None:
                        mask_sample = gather_target_instances(nb.mask, sample_ids, node_ids)
                        eval_masks.append(mask_sample.detach())

        if not targets:
            raise RuntimeError("Evaluation produced no target positions")

        targets = torch.concat(targets, dim=0)
        predicts = torch.concat(predicts, dim=0)
        eval_masks_tensor = torch.concat(eval_masks, dim=0) if eval_masks else None

        mae, rmse, mape, mape_10, mape_20 = cal_metrics(
            predicts=predicts, targets=targets, eval_mask=eval_masks_tensor
        )

    if save and log_dir is not None:
        result = {"targets": targets.cpu().numpy(), "predicts": predicts.cpu().numpy()}
        np.savez(os.path.join(log_dir, "test.npz"), **result)

    return mae, rmse, mape, mape_10, mape_20, target_diag_stats(targets)


# ---------------------------------------------------------------------------
# Bundle evaluation — used by both validation and test
# ---------------------------------------------------------------------------


def eval_bundles(model, bundles, aux_pools_map, make_step_fn, eval_fn, agg, mylogger, tag):
    """Iterate bundles, run eval_fn under each aux-pool set, collect per-bundle results."""
    results = []
    for b in bundles:
        model.set_graph(b["key"])
        step_fn = make_step_fn(b)
        result = eval_fn(model, aux_pools_map[b["key"]], step_fn, agg)
        results.append(result)
    return results
