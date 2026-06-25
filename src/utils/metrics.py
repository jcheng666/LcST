import numpy as np
import torch


def MAE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred))

def MSE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean((pred - true) ** 2)

def RMSE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))


def MAPE_torch(pred, true, mask_value=1e-6):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))

def MAPE_torch_node(pred, true, mask_value=1e-6):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = pred*mask
        true = true*mask + (1-mask.float())
        count = mask.sum(dim=-1)
    return torch.sum(torch.abs(torch.div((true - pred)*mask, true)),dim=-1)/count


def cal_metrics(predicts, targets, eval_mask=None):
    F = targets.shape[-1]

    mae = []
    for f in range(F):
        if eval_mask is not None:
            mask = eval_mask[..., f].bool()
            mae.append(MAE_torch(pred=predicts[..., f][mask], true=targets[..., f][mask]).item())
        else:
            mae.append(MAE_torch(pred=predicts[..., f], true=targets[..., f]).item())

    rmse = []
    for f in range(F):
        if eval_mask is not None:
            mask = eval_mask[..., f].bool()
            rmse.append(RMSE_torch(pred=predicts[..., f][mask], true=targets[..., f][mask]).item())
        else:
            rmse.append(RMSE_torch(pred=predicts[..., f], true=targets[..., f]).item())

    mape = []
    for f in range(F):
        if eval_mask is not None:
            mask = eval_mask[..., f].bool()
            mape.append(MAPE_torch(pred=predicts[..., f][mask], true=targets[..., f][mask]).item())
        else:
            mape.append(MAPE_torch(pred=predicts[..., f], true=targets[..., f]).item())

    mape_10 = []
    for f in range(F):
        if eval_mask is not None:
            mask = eval_mask[..., 0].bool() & (targets[..., 0] >= 10)
            mape_10.append(MAPE_torch(pred=predicts[..., f][mask], true=targets[..., f][mask]).item())
        else:
            mask = targets[..., 0] >= 10
            mape_10.append(MAPE_torch(pred=predicts[..., f][mask], true=targets[..., f][mask]).item())

    mape_20 = []
    for f in range(F):
        if eval_mask is not None:
            mask = eval_mask[..., 0].bool() & (targets[..., 0] >= 20)
            mape_20.append(MAPE_torch(pred=predicts[..., f][mask], true=targets[..., f][mask]).item())
        else:
            mask = targets[..., 0] >= 20
            mape_20.append(MAPE_torch(pred=predicts[..., f][mask], true=targets[..., f][mask]).item())

    return mae, rmse, mape, mape_10, mape_20


def average_metric_lists(metric_lists):
    valid = [values for values in metric_lists if values is not None]
    if not valid:
        return None
    if isinstance(valid[0], (float, int)):
        return float(np.asarray(valid, dtype=np.float64).mean())
    return np.asarray(valid, dtype=np.float64).mean(axis=0).tolist()


def average_eval_metrics(results):
    return tuple(average_metric_lists(values) for values in zip(*results))


def target_diag_stats(targets, sample_limit=262_144):
    values = targets[..., 0].detach().float().reshape(-1)
    if values.numel() == 0:
        return None
    sample_values = values
    if values.numel() > sample_limit:
        sample_ids = torch.arange(sample_limit, device=values.device, dtype=torch.long)
        sample_ids = sample_ids * (values.numel() - 1) // (sample_limit - 1)
        sample_values = values[sample_ids]
    try:
        quantiles = torch.quantile(
            sample_values,
            torch.tensor([0.1, 0.5, 0.9], device=values.device),
        )
        return [
            sample_values.mean().item(),
            sample_values.std(unbiased=False).item(),
            sample_values.min().item(),
            quantiles[0].item(),
            quantiles[1].item(),
            quantiles[2].item(),
        ]
    except RuntimeError:
        return None


def fmt_stats(stats):
    if stats is None:
        return "target_stats:None"
    mean, std, min_value, p10, p50, p90 = stats
    return (
        f"target_mean:{mean} target_std:{std} target_min:{min_value} target_p10:{p10} target_p50:{p50} target_p90:{p90}"
    )


def fmt(v):
    """Format a float to 5 significant decimal places."""
    if isinstance(v, float):
        return f"{v:.5f}"
    return v
