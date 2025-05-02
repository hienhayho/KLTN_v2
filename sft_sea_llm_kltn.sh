#!/bin/bash
#
# 22GB

DATA=$1
OUT=$2
E=${3:-"3"}

HF_HOME="./LLaMA-Factory/model_cache" HF_TOKEN= WANDB_API_KEY= MODELSCOPE_CACHE="./model_cache" NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 \
    swift sft \
    --model SeaLLMs/SeaLLMs-v3-1.5B-Chat \
    --model_type qwen2 \
    --train_type lora \
    --dataset "$DATA" \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --report_to wandb \
    --num_train_epochs "$E" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --deepspeed zero2 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 1 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 50 \
    --max_length 8196 \
    --output_dir "$OUT" \
    --warmup_ratio 0.05 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --use_hf true
