"""Optimizer / loss function / scheduler construction."""

import torch

from utils.muon import Muon


def build_loss_fn(args):
    if args.loss_type == "mae":
        return torch.nn.L1Loss()
    if args.loss_type == "huber":
        return torch.nn.HuberLoss(delta=args.huber_delta)
    raise ValueError(f"unknown loss_type: {args.loss_type}")


def build_optimizer(model, args, mylogger):
    muon_params = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and p.ndim == 2 and "embed" not in name and "basemodel" not in name
    ]

    adam_params = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and (p.ndim != 2 or "embed" in name or "basemodel" in name)
    ]

    mylogger.info(f"[Optimizer] MuonWithAuxAdam muon_params={len(muon_params)} adam_params={len(adam_params)}")
    return Muon(
        muon_params=muon_params,
        adamw_params=adam_params,
        lr=args.lr,
        wd=args.weight_decay,
        adamw_betas=(0.9, 0.95),
    )


def build_scheduler(optim):
    """Build ReduceLROnPlateau scheduler. Returns (scheduler, scheduler_args_dict)."""
    scheduler_args = dict(mode="min", factor=0.1, patience=8, min_lr=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, **scheduler_args)
    return scheduler, scheduler_args
