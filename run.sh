#!/bin/bash

# Use GPU 1 and 2
export CUDA_VISIBLE_DEVICES=1
torchrun --nproc_per_node=1 ./main_dino.py \
    --data_path ../../Dataset/CV/birdclef-2025 \
    --csv_path ../../Dataset/CV/birdclef-2025/train_final_mod.csv \
    --local_crops_number 0 \
    --epoch 100 \
    --warmup_epochs 10 \
    --arch efficientnet_b0 \
    --output_dir ./pretrain_output \
    --batch_size_per_gpu 128 \
    --lr 0.05 \
    --min_lr 0.0005 \
    --warmup_teacher_temp 0.03 \
    --norm_last_layer true \
    --out_dim 8192 \
    --optimizer adamw
    