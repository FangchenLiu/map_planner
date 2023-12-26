# Mapping State Space using Landmarks for Universal Goal Reaching
We introduce a method to combine low-level RL policy with search algorithm.

## Paper
This paper is accepted by NeurIPS 2019. For more details, please refer to this link: 
[http://papers.nips.cc/paper/8469-mapping-state-space-using-landmarks-for-universal-goal-reaching](http://papers.nips.cc/paper/8469-mapping-state-space-using-landmarks-for-universal-goal-reaching).

## Run the code
We provide the scripts for PointMaze and AntMaze. 
```bash
./scripts/train_pointmaze.sh
./scripts/train_antmaze.sh
```
You can also customize your own environments and find suitable parameters.
The environments can be found in ``./goal_env``.

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