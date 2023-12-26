#!/usr/bin/env bash

python train_ddpg.py --env-name PointMaze-v1 --test PointMazeTest-v1 --random-eps 0.01 --device cuda:0 --future-step 80 --gamma 0.98 --n-epochs 7000 --period 10 \
--fps --landmark 350 --initial-sample 2000 --clip-v -4