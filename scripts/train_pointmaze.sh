#!/usr/bin/env bash

python train_ddpg.py --env-name PointMaze-v1 --test PointMazeTest-v1 --random-eps 0.01 --device cuda:0 --future-step 80 --gamma 0.98 --n-epochs 7200 --period 10 \
--fps --landmark 200 --initial-sample 2000 --clip-v -4 --lr-decay-actor 3000 --lr-decay-critic 4000 --resume --resume-path saved_models/PointMaze-v1_Apr06_02-59-22 --resume-epoch 1500