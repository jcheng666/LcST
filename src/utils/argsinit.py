import argparse


def AddModelArgs(parser):
    parser.add_argument("--model", default="transformer", type=str, help="backbone name: gpt2 or transformer")
    parser.add_argument("--lora", action="store_true", help="whether use lora fine-tunning")
    parser.add_argument("--ln_grad", action="store_true", help="whether to calculate gradient of LayerNorm")
    parser.add_argument("--causal", default=1, type=int, help="backbone causal attention")
    parser.add_argument("--llm_layers", default=None, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--unit_len", default=36, type=int)
    parser.add_argument("--token_dim", default=96, type=int)
    parser.add_argument("--token_head", default=2, type=int)
    parser.add_argument("--n_aux", default=16, type=int, help="number of auxiliary nodes per target")
    parser.add_argument(
        "--aux_neighbor_order",
        default="topological",
        choices=["index", "topological"],
        type=str,
        help="neighbor auxiliary ordering before truncation/fill; default: topological",
    )
    parser.add_argument(
        "--aux_neighbor_fill",
        default="higher_order",
        choices=["repeat_1hop", "higher_order"],
        type=str,
        help="repeat 1-hop neighbors or fill with 2-hop then 3-hop neighbors; default: higher_order",
    )
    parser.add_argument("--backbone_capacity", default=4096, type=int,
                        help="maximum target instances sent through the backbone/reasoner at once")
    parser.add_argument("--node_pe_k", default=16, type=int,
                        help="Laplacian PE eigenvectors to use (0 = disabled); adds topological identity to tokens")
    parser.add_argument("--no_node_pe", action="store_true",
                        help="explicitly disable Laplacian PE even when --node_pe_k > 0")
    parser.add_argument("--no_global_token", action="store_true",
                        help="disable global token (mean-pool over all nodes) in reasoner dispatch")
    parser.add_argument("--final_aux_pool_sets", default=3, type=int,
                        help="auxiliary pool sets averaged for the final test at training end")
    parser.add_argument("--routine_aux_pool_sets", default=1, type=int,
                        help="auxiliary pool sets averaged during in-loop validation/test. Must be <= --final_aux_pool_sets.")
    parser.add_argument("--aux_pool_seed", default=2026, type=int, help="dedicated seed for fixed validation/test auxiliary pools")


def AddDataArgs(parser):
    parser.add_argument("--dataset", type=str,
                        help="dataset name(s), comma-separated for multiple")
    parser.add_argument("--data_path", type=str,
                        help="data path(s), comma-separated matching --dataset")
    parser.add_argument("--adj_filename", default=None, type=str,
                        help="adjacency path(s), comma-separated matching --dataset")

    parser.add_argument("--sample_len", default=2016, type=int)
    parser.add_argument("--predict_len", default=12, type=int)
    parser.add_argument("--output_len", default=12, type=int)
    parser.add_argument("--stride", default=12, type=int)
    parser.add_argument("--train_ratio", default=0.6, type=float)
    parser.add_argument("--val_ratio", default=0.2, type=float)
    parser.add_argument("--input_dim", default=1, type=int)
    parser.add_argument("--output_dim", default=None, type=int, help="output feature count; defaults to input_dim")


def AddTrainArgs(parser):
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--lr_warmup_steps", default=1000, type=int,
                        help="linear warmup steps before cosine learning-rate decay")
    parser.add_argument("--min_lr_ratio", default=0.1, type=float,
                        help="final learning-rate ratio for cosine decay")
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--loss_type", default="mae", choices=["mae", "huber"], type=str)
    parser.add_argument("--huber_delta", default=30.0, type=float,
                        help="delta for HuberLoss when --loss_type huber")
    parser.add_argument("--batch_size", default=256, type=int,
                        help="target instances per optimizer step")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="DataLoader worker processes")
    parser.add_argument("--prefetch_factor", default=2, type=int,
                        help="DataLoader prefetch factor when num_workers > 0")
    parser.add_argument("--steps_per_epoch", default=0, type=int,
                        help="train forward steps per epoch; 0=derive from train_target_instances budget")
    parser.add_argument("--train_target_instances", default=1_476_450, type=int,
                        help="target-instance budget per epoch for training")
    parser.add_argument("--eval_instance_budget", default=262_144, type=int,
                        help="window-node instances sampled for each in-loop validation/test")
    parser.add_argument("--eval_budget", default=None, type=int,
                        help="deprecated alias for --eval_instance_budget")
    parser.add_argument("--epoch", default=100, type=int, help="number of train/eval rounds")
    parser.add_argument("--val_every", default=1000, type=int, help="validation interval in train forward steps")
    parser.add_argument("--test_every", default=5000, type=int, help="test interval in train forward steps")
    parser.add_argument("--patience", default=30, type=int)

    parser.add_argument("--log_every", default=100, type=int,
                        help="log training loss every N steps")
    parser.add_argument("--step_limit", default=0, type=int,
                        help="max steps per epoch for smoke validation, 0=full")
    parser.add_argument("--mix_seed", default=2026, type=int,
                        help="seed for random batch mixing across datasets")



def InitArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--desc", default="gpt2_s_token", type=str, help="description")
    parser.add_argument("--log_root", default="../logs", type=str, help="Log root directory")
    parser.add_argument("--from_pretrained_model", default=None, type=str)
    parser.add_argument("--zero_shot", action="store_true")
    parser.add_argument("--save_result", action="store_true")
    parser.add_argument("--few_shot", default=1, type=float)
    parser.add_argument("--node_shuffle_seed", default=None, type=int)
    parser.add_argument("--target_strategy", default="random", choices=["random", "hybrid"], type=str)
    parser.add_argument(
        "--target_mode",
        default="forecast",
        choices=["forecast", "impute_last", "impute_full"],
        type=str,
        help="forecast: use sample_len context to predict output_len future steps; impute_last: use sample_len window and fill its last output_len steps; impute_full: reconstruct entire sample_len window, loss only on mask=0 positions",
    )

    AddDataArgs(parser)
    AddModelArgs(parser)
    AddTrainArgs(parser)

    args = parser.parse_args()
    args.window_batch_size = 8

    if args.routine_aux_pool_sets > args.final_aux_pool_sets:
        raise ValueError(
            f"--routine_aux_pool_sets ({args.routine_aux_pool_sets}) must be <= "
            f"--final_aux_pool_sets ({args.final_aux_pool_sets})"
        )
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.lr_warmup_steps < 0:
        raise ValueError("--lr_warmup_steps must be non-negative")
    if not 0 <= args.min_lr_ratio <= 1:
        raise ValueError("--min_lr_ratio must be in [0, 1]")
    if args.batch_size < 64:
        raise ValueError(
            f"--batch_size={args.batch_size} is too small. "
            "--batch_size is optimizer target-instance count and must be >= 64 for Muon runs."
        )
    if args.huber_delta <= 0:
        raise ValueError("--huber_delta must be positive")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch_factor must be positive")
    if args.steps_per_epoch < 0:
        raise ValueError("--steps_per_epoch must be non-negative")
    if args.val_every <= 0:
        raise ValueError("--val_every must be positive")
    if args.test_every <= 0:
        raise ValueError("--test_every must be positive")
    if args.eval_budget is not None:
        args.eval_instance_budget = args.eval_budget
    if args.eval_instance_budget <= 0:
        raise ValueError("--eval_instance_budget must be positive")
    if args.step_limit < 0:
        raise ValueError("--step_limit must be non-negative")

    if args.backbone_capacity <= 0:
        raise ValueError("--backbone_capacity must be positive")

    if args.output_dim is None:
        args.output_dim = args.input_dim

    return args
