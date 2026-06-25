export CUDA_VISIBLE_DEVICES=0
PYTHON_BIN="/data/HuangYiheng/miniconda3/envs/llm/bin/python"

"${PYTHON_BIN}" main.py \
    --data_path '/data/HuangYiheng/github/STD-PLM/data/traffic/PEMS03/PEMS03.npz' \
    --adj_filename /data/HuangYiheng/github/STD-PLM/data/traffic/PEMS03/PEMS03.csv \
    --dataset PEMS03FLOW \
    --desc PEMS03_pre\
    --sample_len 2016 \
    --predict_len 12 \
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
    --input_dim 1\
    --output_dim 1
