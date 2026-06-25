import os
import random
import string

from logger import getlogger
from model.backbone import build_backbone
from model.model import STALLM
from utils.argsinit import InitArgs
from utils.utils import check_dir, get_time_str
from utils.git_snapshot import snapshot_experiment

from trainer import (
    Train,
    install_model_graphs,
    load_bundles,
)

random_str = lambda: "".join(random.sample(string.ascii_letters + string.digits, 6))


def build_model(args, basemodel, output_len, adj_mx):
    tokenizer_input_dim = args.input_dim
    return STALLM(
        basemodel=basemodel,
        sample_len=args.sample_len,
        output_len=output_len,
        input_dim=tokenizer_input_dim,
        output_dim=args.output_dim,
        adj_mx=adj_mx,
        dropout=args.dropout,
        unit_len=args.unit_len,
        token_dim=args.token_dim,
        token_head=args.token_head,
        n_aux=args.n_aux,
        aux_neighbor_order=args.aux_neighbor_order,
        aux_neighbor_fill=args.aux_neighbor_fill,
        backbone_capacity=args.backbone_capacity,
        node_pe_k=args.node_pe_k,
        node_pe_enabled=not args.no_node_pe,
        no_global_token=args.no_global_token,
    ).cuda()


if __name__ == "__main__":
    args = InitArgs()

    output_len = args.output_len
    window_size = args.sample_len
    if args.target_mode == "forecast":
        window_size = args.sample_len + args.output_len
    elif args.target_mode == "impute_full":
        window_size = args.sample_len

    basemodel = build_backbone(args)

    data_bundles = load_bundles(args, output_len, window_size)
    adj_mx = data_bundles[0]["adj_mx"]

    LOG_DIR = os.path.join(args.log_root, f"{get_time_str()}_{args.desc}_{random_str()}")

    check_dir(LOG_DIR, mkdir=True)
    snapshot_experiment(LOG_DIR, args)

    logpath = os.path.join(LOG_DIR, "experiments.log")
    modelpath = os.path.join(LOG_DIR, f"{get_time_str()}_{args.desc}.pth")

    mylogger = getlogger(logpath)

    mylogger.info(
        f"desc:{args.desc} dataset:{args.dataset} model:{args.model} "
        f"sample:{args.sample_len} output:{args.output_len} batch:{args.batch_size} "
        f"lr:{args.lr} epoch:{args.epoch}"
    )

    model = build_model(args, basemodel, output_len, adj_mx)
    install_model_graphs(model, data_bundles)

    sep_initialized = model.init_sep_from_eos()
    mylogger.info(f"sep_token_initialized_from_eos:{sep_initialized}")

    if args.from_pretrained_model is not None:
        model.load(args.from_pretrained_model)

    if args.zero_shot and args.from_pretrained_model is None:
        mylogger.info("Please specify pretrained model when test zero-shot")
        exit()

    total_params, total_trainable_params = model.params_num()
    mylogger.info(f"total_params:{total_params}    total_trainable_params:{total_trainable_params}")

    Train(args, mylogger, model, data_bundles, log_dir=LOG_DIR, modelpath=modelpath)
