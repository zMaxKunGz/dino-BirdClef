torchrun --nproc_per_node=2 ./main_dino.py \
    --data_path ../../Dataset/CV/birdclef-2025 \
    --csv_path ../../Dataset/CV/birdclef-2025/train_final_mod.csv \
    --local_crops_number 0 \
    --epoch 10 \
    --warmup_epochs 5 \
    --arch efficientnet_b0