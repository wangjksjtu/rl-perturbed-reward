# RL with Perturbed Rewards

This is the tensorflow implementation of [Reinforcement Learning with Perturbed Rewards](https://arxiv.org/abs/1810.01032) as described in the following AAAI 2020 paper (__Spotlight__):

```
@inproceedings{wang2020rlnoisy,
  title={Reinforcement Learning with Perturbed Rewards},
  author={Wang, Jingkang and Liu, Yang and Li, Bo},
  booktitle={AAAI},
  year={2020}
}
```

The implementation is based on [keras-rl](https://github.com/keras-rl/keras-rl) and [OpenAI baselines](https://github.com/openai/baselines) frameworks. Thanks to the original authors!

- `gym-control`: Classic control games
- `gym-atari`:   Atari-2600 games

## Dependencies
- python 3.5
- tensorflow 1.10.0, keras 2.1.0
- gym, scipy, scipy, joblib, keras
- progressbar2, mpi4py, cloudpickle, opencv-python, h5py, pandas

Note: make sure that you have successfully installed the baseline package and other packages following (using [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) to create virtual environment):
```
mkvirtualenv rl-noisy --python==/usr/bin/python3
pip install -r requirements.txt
cd gym-atari/baselines
pip install -e .
```

## Examples
- Classic control (DQN on Cartpole)
```
cd gym-control
python cem_cartpole.py                                           # true reward
python dqn_cartpole.py --error_positive 0.1 --reward noisy       # perturbed reward
python dqn_cartpole.py --error_positive 0.1 --reward surrogate   # surrogate reward (estimated)
```
- Atari-2600 (PPO on Phoenix)
```
cd gym-atari/baselines
python -m baselines.run --alg=ppo2 --env=PhoenixNoFrameskip-v4 \  # true reward
       --num_timesteps=5e7 --normal=True                          
python -m baselines.run --alg=ppo2 --env=PhoenixNoFrameskip-v4 \  # noisy reward
       --num_timesteps=5e7 --save_path=logs-phoenix/phoenix/ppo2_50M_noisy_0.2 \
       --weight=0.2 --normal=False --surrogate=False --noise_type=anti_iden
python -m baselines.run --alg=ppo2 --env=PhoenixNoFrameskip-v4 \  # surrogate reward (estimated)
       --num_timesteps=5e7 --save_path=logs-phoenix/phoenix/ppo2_50M_noisy_0.2 \
       --weight=0.2 --normal=False --surrogate=True --noise_type=anti_iden
```

## Reproduce the Results
To reproduce all the results reported in the paper, please refer to `scripts/` folders in `rl-noisy-reward-control` and `rl-noisy-reward-atari`:
- `gym-control/scripts`
  - Cartpole
    - `train-cem.sh` (CEM)
    - `train-dqn.sh` (DQN)
    - `train-duel-dqn.sh` (Dueling-DQN)
    - `train-qlearn.sh` (Q-Learning)
    - `train-sarsa.sh` (Deep SARSA)
  - Pendulum
    - `train-ddpg.sh` (DDPG)
    - `train-naf.sh` (NAF)
- `gym-atari/scripts`
  - `train-alien.sh` (Alien)
  - `train-carnival.sh` (Carnival)
  - `train-mspacman.sh` (MsPacman)
  - `train-phoenix.sh` (Phoenix)
  - `train-pong.sh` (Pong)
  - `train-seaquest.sh` (Seaquest)
  - `train-normal.sh` (Training with true rewards)


If you have eight available GPUs (Memory > 8GB), you can directly run the `*.sh` scripts one at a time. Otherwise, you can follow the instructions in the scripts and run the experiments. It ususally takes one or two days (GTX-1080 Ti) to train the policy.
```
cd rl-noisy-reward-atari/baselines
sh scripts/train-alien.sh
```
The logs and models will be saved automatically. We provide `results_single.py` for getting the averaged scores:
```
python -m baselines.results_single --log_dir logs-alien
```

## Citation
Please cite our paper if you use this code in your research work.

## Questions/Bugs
Please submit a Github issue or contact wangjk@cs.toronto.edu if you have any questions or find any bugs.
