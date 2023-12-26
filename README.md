# Mapping State Space using Landmarks for Universal Goal Reaching
We combine low-level RL policy with search algorithm to solve the goal-reaching problem. We build a graph-based environment map
from past experience which couples sampling and end-to-end training.

## Paper
This paper is accepted by NeurIPS 2019. For more details, please refer to [our paper](http://papers.nips.cc/paper/8469-mapping-state-space-using-landmarks-for-universal-goal-reaching).

## Run
We provide the scripts for 2DPlane, PointMaze and AntMaze. 
```bash
./scripts/train_2dplane.sh
./scripts/train_pointmaze.sh
./scripts/train_antmaze.sh
```
You can also customize your own environments and find suitable parameters.

The goal reaching environments can be found in ``./goal_env``.

## Cite Our Paper
If you find it useful, please consider to cite our paper.
```
@inproceedings{huang2019mapping,
  title={Mapping state space using landmarks for universal goal reaching},
  author={Huang, Zhiao and Liu, Fangchen and Su, Hao},
  booktitle={Advances in Neural Information Processing Systems},
  pages={1940--1950},
  year={2019}
}
```
## Acknowledgement
- [Openai Baselines](https://github.com/openai/baselines)
- [Pytorch Baseline implemented by TianhongDai](https://github.com/TianhongDai/hindsight-experience-replay)

## Pretrained Model
- The pretrained model for PointMaze with the default architecture can be founded [here](https://drive.google.com/drive/folders/1S00JbuG2KHM5OhGhfwVkyKK0lcny64_M?usp=sharing).
This is trained with 0.4M interaction steps.
- The pretrained model for AntMaze: will upload soon. 