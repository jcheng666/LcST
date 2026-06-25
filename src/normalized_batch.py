"""NormalizedBatch — encapsulates context normalization + scale restoration.

Collapses the separate normalize_observed_context / restore_instance_scale
dance into a single object that holds the normalization state.
"""

import torch


class NormalizedBatch:
    """Holds a normalized model input together with the context statistics
    needed to restore predictions to the original data scale."""

    def __init__(self, input, target, mask, args, device):
        self.args = args
        self.device = device

        input = input.to(device=device, dtype=torch.float32, non_blocking=True)
        target = target.to(device=device, dtype=torch.float32, non_blocking=True)

        self.target = target
        context_mask = None
        if mask is not None:
            context_mask = mask.to(device=device, dtype=torch.float32, non_blocking=True)

        self.model_input, self.context_mean, self.context_std = _normalize(input, context_mask)
        B, T, N, _ = self.model_input.shape
        self.model_input = self.model_input.permute(0, 2, 1, 3).contiguous().view(B, N, -1)

    def restore(self, predict, sample_ids, node_ids):
        return _restore_scale(predict, self.context_mean, self.context_std,
                              sample_ids, node_ids)


def _normalize(input, context_mask):
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


def _restore_scale(predict, context_mean, context_std, sample_ids, node_ids):
    output_dim = predict.shape[-1]
    mean = context_mean[:, :, :, :output_dim][sample_ids, 0, node_ids, :]
    std = context_std[:, :, :, :output_dim][sample_ids, 0, node_ids, :]
    return predict * (std.unsqueeze(1) + 1e-6) + mean.unsqueeze(1)
