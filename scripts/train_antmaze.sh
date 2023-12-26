#!/usr/bin/env bash
python train_ddpg.py --env-name AntMaze-v1 --test AntMazeTest-v1 --device cuda:1 --gamma 0.98 --n-epochs 14000 --period 3 \
--fps --landmark 500 --initial-sample 2000 --clip-v -32