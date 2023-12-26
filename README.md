# Mapping State Space using Landmarks for Universal Goal Reaching
Our method is designed for goal-reaching problem, in which we combine low-level RL policy with search on a graph-based environment map.

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
- Will upload soon