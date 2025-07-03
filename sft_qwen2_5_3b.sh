#!/bin/bash
#
# 22GB

# Qwen/Qwen2.5-3B-Instruct

MODEL=$1
DATA=$2
OUT=$3
E=${4:-"3"}

HF_HOME="./LLaMA-Factory/model_cache" MODELSCOPE_CACHE="./model_cache" NPROC_PER_NODE=1 \
    swift sft \
    --model "$MODEL" \
    --train_type lora \
    --dataset "$DATA" \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --report_to wandb \
    --num_train_epochs "$E" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --lora_rank 8 \
    --deepspeed zero2 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --gradient_accumulation_steps 1 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 50 \
    --max_length 9000 \
    --output_dir "$OUT" \
    --warmup_ratio 0.05 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 2 \
    --use_hf true
# --use_liger_kernel true

# swift export \
#     --adapter ".../checkpoint-xxx/" \
#     --output_dir "./model_output" \
#     --use_hf true \
#     --merge_lora true

# NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 \  # Chạy trên 1 GPU cụ thể (GPU số 0)
#     swift sft \  # Gọi lệnh fine-tuning SFT (Supervised Fine-Tuning) từ framework Swift
#         --model Qwen/Qwen2.5-1.5B-Instruct \  # Model base: Qwen2.5-1.5B-Instruct
#         --train_type lora \  # Sử dụng kỹ thuật fine-tuning LoRA (Low-Rank Adaptation)
#         --dataset "..." \  # Tên dataset dùng để huấn luyện (nên thay thế bằng tên cụ thể)
#         --split_dataset_ratio 0 \  # Không tách tập huấn luyện và đánh giá (dùng toàn bộ cho training)
#         --torch_dtype bfloat16 \  # Sử dụng bfloat16 giúp tiết kiệm bộ nhớ mà vẫn đảm bảo độ chính xác
#         --report_to wandb \  # Log kết quả huấn luyện lên Weights & Biases (wandb)
#         --num_train_epochs 1 \  # Số epoch huấn luyện là 1
#         --per_device_train_batch_size 1 \  # Batch size huấn luyện trên mỗi GPU
#         --per_device_eval_batch_size 1 \  # Batch size đánh giá trên mỗi GPU
#         --learning_rate 1e-5 \  # Tốc độ học (learning rate)
#         --lora_rank 8 \  # Rank trong LoRA (quyết định số lượng tham số cần huấn luyện)
#         --deepspeed zero2 \  # Sử dụng DeepSpeed optimization stage 2 (Zero Redundancy Optimizer stage 2)
#         --lora_alpha 16 \  # Hệ số alpha dùng để scale trong LoRA
#         --target_modules all-linear \  # Áp dụng LoRA vào tất cả các lớp linear trong model
#         --gradient_accumulation_steps 1 \  # Tích lũy gradient mỗi 1 bước (không tích lũy thêm)
#         --save_steps 1000 \  # Lưu checkpoint mỗi 1000 bước
#         --save_total_limit 2 \  # Chỉ giữ lại 2 checkpoint gần nhất để tiết kiệm ổ đĩa
#         --logging_steps 50 \  # Log loss/metrics sau mỗi 50 bước
#         --max_length 9000 \  # Độ dài tối đa của input sequence (áp dụng với tokenizer + truncation)
#         --output_dir "outputs" \  # Thư mục lưu checkpoint và log đầu ra
#         --warmup_ratio 0.05 \  # Phần trăm warm-up (5% đầu tiên tăng dần learning rate)
#         --dataset_num_proc 4 \  # Sử dụng 4 process để tiền xử lý dataset song song
#         --dataloader_num_workers 2 \  # Số lượng worker cho DataLoader khi huấn luyện
#         --use_hf true  # Dùng Hugging Face (do mặc định ms-swift dùng ModelScope)
