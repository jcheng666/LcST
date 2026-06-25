CONDA_SH="/data/HuangYiheng/miniconda3/etc/profile.d/conda.sh"
source "${CONDA_SH}"
conda activate llm
PYTHON_BIN="$(which python)"

export CUDA_VISIBLE_DEVICES=4

ADJ_PATH='/data/HuangYiheng/LLM/data/traffic/PEMS08/PEMS08.csv'
# ===== SCTC =====
DATA_PATH='/data/HuangYiheng/LLM/data/traffic/miss_data/PEMS08/true_data_SC-TC_0.7_v2.npz'
DESC='PEMS08_IM_SCTC70_2016_12'
# ===== SRTR =====
# DATA_PATH='/data/HuangYiheng/LLM/data/traffic/miss_data/PEMS08/true_data_SR-TR_0.7_v2.npz'
# DESC='PEMS08_IM_SRTR70_2016_12'

"${PYTHON_BIN}" main.py \
    --dataset PEMS08MISSING \
    --data_path "${DATA_PATH}" \
    --adj_filename "${ADJ_PATH}" \
    --save_result \
    --desc "${DESC}" \
    --sample_len 2016 \
    --predict_len 12 \
    --output_len 12 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --epoch 500 \
    --batch_size 64\
    --lr 0.001 \
    --causal 0 \
    --model gpt2 \
    --patience 30 \
    --ln_grad \
    --lora \
    --llm_layers 3 \
    --dropout 0.05 \
    --weight_decay 0 \
    --input_dim 1 \
    --output_dim 1
