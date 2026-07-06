#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python train_model.py \
    --epoch 20 \
    --lr 2e-4 \
    --batch_size 1 \
    --num_worker 8 \
    --dataset gopro \
    --voxel_bins 128 \
    --nb_of_flow 16 \
    --save_dir train \
    --seed 1234 \
    "$@"
