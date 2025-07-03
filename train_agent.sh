#!/bin/bash

HF_HOME="./LLaMA-Factory/model_cache" MODELSCOPE_CACHE="./model_cache" NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 \
    swift sft \
    --model Qwen/Qwen2.5-3B-Instruct \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --dataset processed_data_full.jsonl \
    --agent_template hermes \
    --torch_dtype bfloat16 \
    --split_dataset_ratio 0 \
    --report_to wandb \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 8 \
    --save_steps 500 \
    --save_total_limit 2 \
    --attn_impl flash_attn \
    --logging_steps 50 \
    --max_length 6000 \
    --save_only_model true \
    --deepspeed zero2 \
    --output_dir sft_model_f/agent_3B-5-5_data_full \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 2 \
    --dataset_num_proc 8 \
    --use_hf true \
    --temperature 0 \
    --use_liger_kernel true
