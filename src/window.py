import torch


def compute_loss(predict, target, loss_fn):
    return loss_fn(predict, target)


def normalize_observed_context(input, context_mask=None):
    if context_mask is None:
        context_mean = input.mean(dim=1, keepdim=True)
        variance = ((input - context_mean) ** 2).mean(dim=1, keepdim=True)
        context_std = torch.sqrt(variance + 1e-6)
        normalized = (input - context_mean) / (context_std + 1e-6)
        return normalized, context_mean, context_std

    observed = context_mask.to(dtype=input.dtype)
    count = observed.sum(dim=1, keepdim=True).clamp(min=1.0)
    context_mean = (input * observed).sum(dim=1, keepdim=True) / count
    variance = ((input - context_mean) ** 2 * observed).sum(dim=1, keepdim=True) / count
    context_std = torch.sqrt(variance + 1e-6)
    normalized = (input - context_mean) / (context_std + 1e-6)
    normalized = torch.where(context_mask == 0, 0, normalized)
    return normalized, context_mean, context_std


def restore_instance_scale(predict, context_mean, context_std, sample_ids, node_ids):
    output_dim = predict.shape[-1]
    mean = context_mean[:, :, :, :output_dim][sample_ids, 0, node_ids, :]
    std = context_std[:, :, :, :output_dim][sample_ids, 0, node_ids, :]
    return predict * (std.unsqueeze(1) + 1e-6) + mean.unsqueeze(1)


def target_instance_chunks(window_count, node_count, batch_size, device, full):
    pool_size = window_count * node_count
    if full:
        flat_ids = torch.arange(pool_size, device=device)
    elif batch_size >= pool_size:
        flat_ids = torch.arange(pool_size, device=device)
    else:
        flat_ids = torch.randperm(pool_size, device=device)[:batch_size]

    for start in range(0, flat_ids.numel(), batch_size):
        chunk = flat_ids[start : start + batch_size]
        yield chunk // node_count, chunk % node_count


def gather_target_instances(values, sample_ids, node_ids):
    if values.dim() == 3:
        return values[sample_ids, :, node_ids].unsqueeze(-1)
    if values.dim() == 4:
        return values[sample_ids, :, node_ids, :]
    raise ValueError(f"target values must be 3D or 4D, got shape {tuple(values.shape)}")
