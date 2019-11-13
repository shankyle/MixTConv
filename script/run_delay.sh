#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=4,5,6,7 python ../main.py somethingv1 RGB \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.01 --lr_steps 35 45 50 --epochs 55 \
#     --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#     --operations=group1dk5 --n_div=8 --npb
