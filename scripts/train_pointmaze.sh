#!/usr/bin/env bash

python train_ddpg.py --env-name PointMaze-v1 --test PointMazeTest-v1 --device cuda:0 --gamma 0.98 --n-epochs 7000 --period 10 \
--fps --landmark 200 --initial-sample 1000 --clip-v -4