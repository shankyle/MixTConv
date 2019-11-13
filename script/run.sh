#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=6,7 python ../main.py somethingv1 RGB \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.01 --lr_steps 30 40 45 --epochs 50 \
#     --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#     --operations=ms_cfm --n_div=4 --npb

#CUDA_VISIBLE_DEVICES=6,7 python ../main.py somethingv1 RGB \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.01 --lr_steps 30 40 45 --epochs 50 \
#     --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#     --operations=ms_group1d --n_div=4 --npb

#CUDA_VISIBLE_DEVICES=4,5 python ../main.py somethingv1 RGB \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.01 --lr_steps 30 40 45 --epochs 50 \
#     --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#     --operations=ms_group1d2x --n_div=4 --npb

#CUDA_VISIBLE_DEVICES=6,7 python ../main.py somethingv1 RGB \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.01 --lr_steps 30 40 45 --epochs 50 \
#     --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#     --operations=ms_ceneargroup1d2x --n_div=8 --npb

#CUDA_VISIBLE_DEVICES=4,5,6,7 python ../main.py somethingv1 RGB \
##     --arch resnet50 --num_segments 8 \
##     --gd 20 --lr 0.01 --lr_steps 30 40 45 --epochs 50 \
##     --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
##     --operations=ms_group1douter --n_div=8 --npb

#CUDA_VISIBLE_DEVICES=0,1,2,3 python ../main.py somethingv1 RGB \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.01 --lr_steps 30 40 45 --epochs 50 \
#     --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#     --operations=ms_ceneargroup1diner --n_div=8 --npb

#CUDA_VISIBLE_DEVICES=4,5 python ../main.py somethingv1 RGB \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.01 --lr_steps 30 40 45 --epochs 50 \
#     --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#     --operations=ms_cenearres --n_div=8 --npb

#CUDA_VISIBLE_DEVICES=8,9 python ../main.py somethingv1 RGB \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.01 --lr_steps 30 40 45 --epochs 50 \
#     --batch-size 58 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#     --operations=ms_corr_cascade --n_div=8 --npb

#CUDA_VISIBLE_DEVICES=0,1,2,3 python ../main.py somethingv1 RGB \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.01 --lr_steps 30 40 45 --epochs 50 \
#     --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#     --operations=ms_groupouter_nl --n_div=8 --npb --non_local

#CUDA_VISIBLE_DEVICES=4,5,6,7 python ../main.py somethingv1 RGB \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.01 --lr_steps 35 45 50 --epochs 55 \
#     --batch-size 32 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#     --operations=ms_corr_cascade --n_div=8 --npb --dwise

#CUDA_VISIBLE_DEVICES=4,5,6,7 python ../main.py somethingv1 RGB \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.01 --lr_steps 35 45 50 --epochs 55 \
#     --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#     --operations=ms_corr --n_div=8 --npb

#CUDA_VISIBLE_DEVICES=8,9 python ../main.py somethingv1 RGB \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.01 --lr_steps 30 40 45 --epochs 50 \
#     --batch-size 64 -j 16 --dropout 0.7 --consensus_type=avg --eval-freq=1 \
#     --operations=stm --n_div=4 --npb --inplace

#CUDA_VISIBLE_DEVICES=0,1,2,3 python ../main.py somethingv1 RGB \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.01 --lr_steps 30 40 45 --epochs 50 \
#     --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#     --operations=group1dsplit --n_div=8 --npb

#CUDA_VISIBLE_DEVICES=4,5,6,7 python ../main.py somethingv1 RGB \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.01 --lr_type earlyadjac --epochs 150 \
#     --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#     --operations=ms_group1douter --n_div=8 --npb

#CUDA_VISIBLE_DEVICES=8,9 python ../main.py somethingv1 RGB \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.01 --lr_type earlyadjac --epochs 150 \
#     --batch-size 64 -j 16 --dropout 0.8 --consensus_type=identity --eval-freq=1 \
#     --operations=ms_group1douter --n_div=8 --npb

#CUDA_VISIBLE_DEVICES=5,6,7 python ../main.py somethingv1 RGB \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.01 --lr_steps 30 40 45 --epochs 50 \
#     --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#     --operations=msgroup1dsplit --n_div=8 --npb

#155
#CUDA_VISIBLE_DEVICES=6,7 python ../main.py somethingv1 RGB \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.01 --lr_steps 40 50 --epochs 60 \
#     --batch-size 24 -j 16 --dropout 0.3 --consensus_type=avg --eval-freq=1 \
#     --operations=gst --n_div=4 --npb

#CUDA_VISIBLE_DEVICES=2,4 python ../main.py somethingv1 RGB \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.01 --lr_steps 30 40 45 --epochs 50 \
#     --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#     --operations=group1dk3outer --n_div=8 --npb

#CUDA_VISIBLE_DEVICES=3,4 python ../main.py somethingv1 RGB \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.01 --lr_steps 30 40 45 --epochs 50 \
#     --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#     --operations=group1dk3pcbam --n_div=8 --npb

#CUDA_VISIBLE_DEVICES=8,9 python ../main.py somethingv1 RGB \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.01 --lr_steps 30 40 45 --epochs 50 \
#     --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
#     --operations=group1dsplitdiv1x1 --n_div=8 --npb

#CUDA_VISIBLE_DEVICES=5,6 python ../main.py somethingv1 RGB \
#     --arch resnet50 --num_segments 8 \
#     --gd 20 --lr 0.01 --lr_steps 30 40 45 --epochs 50 \
#     --batch-size 64 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
#     --operations=group1dsplitdiv --n_div=8 --npb --wd 1e-4

#CUDA_VISIBLE_DEVICES=0,1,2,3 python ../main.py somethingv1 RGB \
#     --arch resnet50 --num_segments 16 \
#     --gd 20 --lr 0.01 --lr_steps 31 40 45 --epochs 50 \
#     --batch-size 64 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
#     --operations=ms_group1douter --n_div=8 --npb --wd 1e-4

CUDA_VISIBLE_DEVICES=5,6 python ../main.py somethingv1 RGB \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.01 --lr_steps 30 40 45 --epochs 50 \
     --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
     --operations=msgroup1dpartalsplitdiv --n_div=8 --npb