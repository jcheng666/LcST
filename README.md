# LcST

Source code and experiment scripts for the paper submitted to PVLDB Vol. XX.

## Repository Structure

```
src/           Main source code (model, trainer, data pipeline, utilities)
scripts/       Shell scripts for launching experiments
pyproject.toml Project configuration and dependencies
```

## Hardware & Software Requirements

- Python ≥ 3.10 with CUDA-capable GPU (≥ 8 GB VRAM recommended)
- Dependencies managed via [uv](https://github.com/astral-sh/uv) (`uv.lock`)
- Tested on Ubuntu 22.04, CUDA 12.x / 13.0, NVIDIA A40 (48 GB)

## Setup

```bash
# Clone the repository
git clone https://github.com/jcheng666/LcST.git
cd LcST

# Install dependencies
uv sync
```

## Reproducing Experiments

All experiments are launched via `src/main.py`. Key parameters are documented
in `src/utils/argsinit.py`.

### Example: PEMS03 Forecasting

```bash
uv run python src/main.py \
  --desc pems03_gpt2_lora \
  --dataset PEMS03FLOW \
  --data_path /path/to/pems03.npz \
  --adj_filename /path/to/pems03_adj.csv \
  --model gpt2 --lora \
  --sample_len 2016 --output_len 12 \
  --unit_len 36 \
  --n_aux 16 --aux_neighbor_order topological --aux_neighbor_fill higher_order \
  --epoch 100 --batch_size 64 \
  --log_root ./logs
```

Adjust `--data_path` and `--adj_filename` to point to your local dataset
copies.  The datasets (PEMS03/04/07/08) are publicly available from
[PeMS](https://pems.dot.ca.gov/).

### Experiment Logs

Each run creates a timestamped directory under `--log_root` containing:

| File                  | Content                                     |
|-----------------------|---------------------------------------------|
| `experiments.log`     | Full training log with per-step metrics     |
| `snapshot/args.json`  | Exact arguments used for the run            |
| `snapshot/HEAD`       | Git commit hash at launch time              |
| `snapshot/command.txt`| Exact shell command invoked                 |
| `metrics.json`        | Final test metrics (MAE, RMSE, MAPE, etc.)  |
| `loss.png`            | Training and validation loss curves         |

## License

This project is released for academic use. See the paper for details.
