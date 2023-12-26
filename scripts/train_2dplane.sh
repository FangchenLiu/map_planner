#!/usr/bin/env bash

python train_ddpg.py --env-name GoalPlane-v0 --test GoalPlaneTest-v0 --device cuda:0 --gamma 0.99 --n-epochs 7000 --period 10 \
--fps --landmark 100 --initial-sample 500 --clip-v -4