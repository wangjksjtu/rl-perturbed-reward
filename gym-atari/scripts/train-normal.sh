cd baselines

(export CUDA_VISIBLE_DEVICES=0 && python -m baselines.run --alg=ppo2 --env=AlienNoFrameskip-v4 --num_timesteps=5e7 --save_path=logs-normal/pong/ppo2_50M_normal --normal=True)&
(export CUDA_VISIBLE_DEVICES=1 && python -m baselines.run --alg=ppo2 --env=CarnivalNoFrameskip-v4 --num_timesteps=5e7 --save_path=logs-normal/carnival/ppo2_50M_normal --normal=True)&
(export CUDA_VISIBLE_DEVICES=2 && python -m baselines.run --alg=ppo2 --env=MsPacmanNoFrameskip-v4 --num_timesteps=5e7 --save_path=logs-normal/mspacman/ppo2_50M_normal --normal=True)&
(export CUDA_VISIBLE_DEVICES=3 && python -m baselines.run --alg=ppo2 --env=PhoenixNoFrameskip-v4 --num_timesteps=5e7 --save_path=logs-normal/phoenix/ppo2_50M_normal --normal=True)&
(export CUDA_VISIBLE_DEVICES=4 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=5e7 --save_path=logs-normal/pong/ppo2_50M_normal --normal=True)&
(export CUDA_VISIBLE_DEVICES=5 && python -m baselines.run --alg=ppo2 --env=SeaquestNoFrameskip-v4 --num_timesteps=5e7 --save_path=logs-normal/seaquest/ppo2_50M_normal --normal=True)&

cd ..
