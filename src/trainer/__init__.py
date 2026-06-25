"""Trainer: coordinating class, public entry points, and data-bundle management."""

import random
import time

from data.data import load_data
from trainer.checkpoint import CheckpointManager
from trainer.epoch import (
    TestEpoch,
    TrainEpoch,
    eval_aux,
    eval_bundles,
    eval_chunk_size,
    routine_aux_pool_sets,
    train_one_step,
    train_steps_per_epoch,
)
from trainer.optim import build_loss_fn, build_optimizer, build_scheduler
from trainer.report import MetricsReporter
from utils.metrics import average_eval_metrics, average_metric_lists, fmt

# ---------------------------------------------------------------------------
# Data-bundle management
# ---------------------------------------------------------------------------


def parse_multi_values(value, name):
    if value is None:
        raise ValueError(f"{name} is required")
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} is empty")
    return values


def load_bundle(dataset, data_path, adj_path, args, output_len, window_size):
    (
        cur_train_loader,
        cur_val_loader,
        cur_test_loader,
        node_num,
        features,
        adj_mx,
        distance_mx,
    ) = load_data(
        dataset=dataset,
        sample_len=args.sample_len,
        output_len=output_len,
        window_size=window_size,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        data_path=data_path,
        adj_path=adj_path,
        target_strategy=args.target_strategy,
        batch_size=args.window_batch_size,
        few_shot=args.few_shot,
        node_shuffle_seed=args.node_shuffle_seed,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        target_mode=args.target_mode,
    )
    return {
        "key": dataset,
        "train_loader": cur_train_loader,
        "val_loader": cur_val_loader,
        "test_loader": cur_test_loader,
        "node_num": node_num,
        "features": features,
        "adj_mx": adj_mx,
        "distance_mx": distance_mx,
    }


def load_bundles(args, output_len, window_size):
    datasets = parse_multi_values(args.dataset, "--dataset")
    data_paths = parse_multi_values(args.data_path, "--data_path")
    adj_paths = parse_multi_values(args.adj_filename, "--adj_filename")
    if not (len(datasets) == len(data_paths) == len(adj_paths)):
        raise ValueError(
            "dataset arguments must have the same length: "
            f"datasets={len(datasets)} data_paths={len(data_paths)} adj_paths={len(adj_paths)}"
        )
    return [
        load_bundle(dataset, data_path, adj_path, args, output_len, window_size)
        for dataset, data_path, adj_path in zip(datasets, data_paths, adj_paths)
    ]


def install_model_graphs(model, data_bundles):
    first_key = data_bundles[0]["key"]
    model.set_graph(first_key, data_bundles[0]["adj_mx"])
    for bundle in data_bundles[1:]:
        model.add_graph(bundle["key"], bundle["adj_mx"])
    model.set_graph(first_key)


def bundle_aux_pool_sets(model, data_bundles, args):
    final_aux_pool_set_count = max(1, args.final_aux_pool_sets)
    aux_pool_sets_by_key = {}
    for idx, bundle in enumerate(data_bundles):
        model.set_graph(bundle["key"])
        aux_pool_sets_by_key[bundle["key"]] = model.sample_aux_pool_sets(
            n_sets=final_aux_pool_set_count,
            seed=args.aux_pool_seed + idx,
        )
    return aux_pool_sets_by_key


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:

    def __init__(self, args, mylogger, model, data_bundles, log_dir):
        self.args = args
        self.mylogger = mylogger
        self.model = model
        self.data_bundles = data_bundles
        self.log_dir = log_dir

        self.epoch_count = args.epoch
        if args.zero_shot:
            self.epoch_count = 0
        self.mix_rng = random.Random(args.mix_seed)
        self.steps_per_epoch = train_steps_per_epoch(args, data_bundles)
        step_limit = args.step_limit if args.step_limit > 0 else None
        if step_limit is not None:
            self.steps_per_epoch = min(self.steps_per_epoch, step_limit)
        self.total_steps = self.epoch_count * self.steps_per_epoch

        self.aux_pool_sets_by_key = bundle_aux_pool_sets(model, data_bundles, args)
        self.train_states = [{"loader_iter": iter(bundle["train_loader"])} for bundle in data_bundles]

    # -- helpers ------------------------------------------------------------

    def _routine_aux(self):
        return {b["key"]: routine_aux_pool_sets(self.aux_pool_sets_by_key[b["key"]], self.args)
                for b in self.data_bundles}

    # -- main loop ----------------------------------------------------------

    def run(self, modelpath=None):
        args = self.args
        mylogger = self.mylogger
        model = self.model
        bundles = self.data_bundles

        optim = build_optimizer(model, args, mylogger)
        scheduler, scheduler_args = build_scheduler(optim)
        loss_fn = build_loss_fn(args)
        loss_desc = f"[Loss] type:{args.loss_type}"
        if args.loss_type == "huber":
            loss_desc += f" delta:{args.huber_delta}"
        mylogger.info(loss_desc)

        checkpoint = CheckpointManager(model, args.patience)
        reporter = MetricsReporter(self.log_dir, mylogger)

        mylogger.info(
            f"datasets:{','.join(b['key'] for b in bundles)} "
            f"window_batch:{args.window_batch_size} target_batch:{args.batch_size} "
            f"steps_per_epoch:{self.steps_per_epoch} "
            f"eval_chunk:{eval_chunk_size(args)} "
            f"eval_window_node_instances:{args.eval_instance_budget} final_test:full "
            f"num_workers:{args.num_workers}"
        )
        mylogger.info(f"train_mix:shuffle_schedule seed:{args.mix_seed}")
        mylogger.info(
            f"[AuxPools] order:{args.aux_neighbor_order} fill:{args.aux_neighbor_fill} n_aux:{args.n_aux} "
            f"routine_sets:{args.routine_aux_pool_sets} final_sets:{max(1, args.final_aux_pool_sets)} "
            f"seed_base:{args.aux_pool_seed}"
        )
        scheduler_desc = " ".join(f"{key}:{value}" for key, value in scheduler_args.items())
        mylogger.info(f"[Scheduler] ReduceLROnPlateau {scheduler_desc}")
        mylogger.info(f"[EarlyStop] patience:{args.patience}")

        last_log_time = time.time()
        recent_losses = []
        for step_id in range(1, self.total_steps + 1):
            if (step_id - 1) % self.steps_per_epoch == 0:
                for b in bundles:
                    model.set_graph(b["key"])
                    model.resample_aux_pools()

            _, train_loss = train_one_step(
                bundles, self.train_states, model, optim, loss_fn, args, self.mix_rng,
            )
            if train_loss is not None:
                recent_losses.append(train_loss)
            if step_id % args.log_every == 0:
                now = time.time()
                speed = args.log_every / max(now - last_log_time, 1e-6)
                last_log_time = now
                train_loss = average_metric_lists(recent_losses)
                recent_losses = []
                reporter.record_train(step_id, train_loss)
                mylogger.info(
                    f"[Train] step {step_id}/{self.total_steps} "
                    f"loss:{fmt(train_loss)} lr:{optim.param_groups[0]['lr']:.2e} it/s:{speed:.2f}"
                )

            # -- validation --
            if step_id % args.val_every == 0:
                val_losses = eval_bundles(
                    model, bundles, self._routine_aux(),
                    make_step_fn=lambda b: lambda: TrainEpoch(
                        b["val_loader"], model, optim, loss_fn, args,
                        is_training=False, resample=False,
                        eval_instance_budget=args.eval_instance_budget,
                    ),
                    eval_fn=eval_aux, agg=average_metric_lists,
                    mylogger=mylogger, tag="",
                )
                val_loss = average_metric_lists(val_losses)
                if val_loss is None:
                    raise RuntimeError("Validation produced no aggregate loss")
                reporter.record_val(step_id, val_loss)
                checkpoint.update(val_loss, model)
                mylogger.info(f"[Validation] step {step_id}/{self.total_steps} val_loss:{fmt(val_loss)}")
                scheduler.step(val_loss)

            # -- periodic test --
            if step_id % args.test_every == 0:
                test_results = eval_bundles(
                    model, bundles, self._routine_aux(),
                    make_step_fn=lambda b: lambda: TestEpoch(
                        b["test_loader"], model, args,
                        resample=False, eval_instance_budget=args.eval_instance_budget,
                    ),
                    eval_fn=eval_aux, agg=average_eval_metrics,
                    mylogger=mylogger, tag="[Test]",
                )
                for b, (mae, rmse, mape, mape_10, mape_20, target_diag_stats) in zip(bundles, test_results):
                    mylogger.info(
                        f"[Test] step {step_id}/{self.total_steps} {b['key']} "
                        f"mae:{fmt(mae)} rmse:{fmt(rmse)} mape:{fmt(mape)}"
                    )

            if checkpoint.should_stop():
                mylogger.info("early stop")
                break

        # -- finalize --
        checkpoint.restore(model)
        if modelpath is not None:
            checkpoint.save(model, modelpath)
            mylogger.info(f"[Model] saved best model:{modelpath}")

        # -- final test --
        final_results = eval_bundles(
            model, bundles, self.aux_pool_sets_by_key,
            make_step_fn=lambda b: lambda: TestEpoch(
                b["test_loader"], model, args,
                resample=False, full_nodes=True,
            ),
            eval_fn=eval_aux, agg=average_eval_metrics,
            mylogger=mylogger, tag="[FinalFull]",
        )

        full_metrics = {}
        for b, (mae, rmse, mape, mape_10, mape_20, target_stats) in zip(bundles, final_results):
            full_metrics[b["key"]] = {
                "mae": mae, "rmse": rmse, "mape": mape,
                "mape_10": mape_10, "mape_20": mape_20, "target_stats": target_stats,
            }

        avg = average_eval_metrics(final_results)
        mae, rmse, mape, mape_10, mape_20, target_stats = avg
        mylogger.info(f"[FinalFull][Average] mae:{fmt(mae)} rmse:{fmt(rmse)} mape:{fmt(mape)}")
        full_metrics["average"] = {
            "mae": mae, "rmse": rmse, "mape": mape,
            "mape_10": mape_10, "mape_20": mape_20, "target_stats": target_stats,
        }
        full_metrics["desc"] = args.desc

        reporter.save_metrics_json(full_metrics)
        if args.save_result:
            reporter.save_npz_results(bundles, model, self.aux_pool_sets_by_key, args)
        reporter.draw_chart()
        return final_results


# ---------------------------------------------------------------------------
# Backward-compatible entry point
# ---------------------------------------------------------------------------


def Train(args, mylogger, model, data_bundles, log_dir, modelpath=None):
    trainer = Trainer(args, mylogger, model, data_bundles, log_dir)
    return trainer.run(modelpath)
