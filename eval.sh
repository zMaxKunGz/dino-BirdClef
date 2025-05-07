#!/bin/bash

# Use GPU 1 and 2
export CUDA_VISIBLE_DEVICES=1
torchrun --nproc_per_node=1 ./eval_linear.py \
    --data_path ../../Dataset/CV/birdclef-2025 \
    --csv_path ../../Dataset/CV/birdclef-2025/train_final_mod.csv \
    --eval_path ../../Dataset/CV/birdclef-2025/test_final_mod.csv \
    --label_path ../../Dataset/CV/birdclef-2025/taxonomy.csv \
    --epoch 10 \
    --arch efficientnet_b0 \
    --output_dir ./eval_output \
    --pretrained_weights ./checkpoint.pth \
    --num_labels 206 \
    --batch_size_per_gpu 128