#!/usr/bin/env bash

python train_ddpg.py --env-name PointMaze-v1 --test PointMazeTest-v1 --random-eps 0.1 --device cuda:0 --future-step 80 --gamma 0.985 --n-epochs 7000 --period 6 \
--fps --landmark 200 --initial-sample 2000 --clip-v -4