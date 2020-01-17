# gym-atari
Reinforcement Learning with Perturbed Reward (Atari Games)

## Usage
### Training
To train models with different noisy or surrogate rewards:
```
sh scripts/train-pong.sh     (Pong-v4)
sh scripts/train-breakout.sh (Breakout-v4)
```
If you want to train the models with specific hyper-parameters by yourself:
```
cd baselines
python -m baselines.run --alg=<name of the algorithm> --env=<environment name> [additional arguments]
```
#### Example 1. PPO with Pong
```
python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 \
                        --save_path=logs-pong/pong/anti_iden/ppo2_30M_noisy_0.1 --weight=0.1 \
                        --normal=False --surrogate=False --noise_type=anti_iden
```

### Visualizing
```
cd baselines
python ../scripts/visualize.py --env_name Breakout --log_dir logs-breakout/ --num_timesteps 50000000 --noise_type anti_iden --all True
```
To see HELP for the visualizing script:
```
python ../scripts/visualize.py -h
```

## References
1. *Proximal Policy Optimization Algorithms* John Schulman et al., 2017
