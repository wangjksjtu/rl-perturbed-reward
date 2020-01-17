cd baselines

(export CUDA_VISIBLE_DEVICES=0 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/anti_iden/ppo2_30M_noisy_0.1 --weight=0.1 --normal=False --surrogate=False --noise_type=anti_iden)&
(export CUDA_VISIBLE_DEVICES=1 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/anti_iden/ppo2_30M_noisy_0.2 --weight=0.2 --normal=False --surrogate=False --noise_type=anti_iden)&
(export CUDA_VISIBLE_DEVICES=2 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/anti_iden/ppo2_30M_noisy_0.3 --weight=0.3 --normal=False --surrogate=False --noise_type=anti_iden)&
(export CUDA_VISIBLE_DEVICES=3 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/anti_iden/ppo2_30M_noisy_0.4 --weight=0.4 --normal=False --surrogate=False --noise_type=anti_iden)&
(export CUDA_VISIBLE_DEVICES=4 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/anti_iden/ppo2_30M_noisy_0.6 --weight=0.6 --normal=False --surrogate=False --noise_type=anti_iden)&
(export CUDA_VISIBLE_DEVICES=5 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/anti_iden/ppo2_30M_noisy_0.7 --weight=0.7 --normal=False --surrogate=False --noise_type=anti_iden)&
(export CUDA_VISIBLE_DEVICES=6 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/anti_iden/ppo2_30M_noisy_0.8 --weight=0.8 --normal=False --surrogate=False --noise_type=anti_iden)&
(export CUDA_VISIBLE_DEVICES=7 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/anti_iden/ppo2_30M_noisy_0.9 --weight=0.9 --normal=False --surrogate=False --noise_type=anti_iden)&

(export CUDA_VISIBLE_DEVICES=0 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/anti_iden/ppo2_30M_surrogate_0.1 --weight=0.1 --normal=False --surrogate=True --noise_type=anti_iden)&
(export CUDA_VISIBLE_DEVICES=1 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/anti_iden/ppo2_30M_surrogate_0.2 --weight=0.2 --normal=False --surrogate=True --noise_type=anti_iden)&
(export CUDA_VISIBLE_DEVICES=2 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/anti_iden/ppo2_30M_surrogate_0.3 --weight=0.3 --normal=False --surrogate=True --noise_type=anti_iden)&
(export CUDA_VISIBLE_DEVICES=3 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/anti_iden/ppo2_30M_surrogate_0.4 --weight=0.4 --normal=False --surrogate=True --noise_type=anti_iden)&
(export CUDA_VISIBLE_DEVICES=4 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/anti_iden/ppo2_30M_surrogate_0.6 --weight=0.6 --normal=False --surrogate=True --noise_type=anti_iden)&
(export CUDA_VISIBLE_DEVICES=5 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/anti_iden/ppo2_30M_surrogate_0.7 --weight=0.7 --normal=False --surrogate=True --noise_type=anti_iden)&
(export CUDA_VISIBLE_DEVICES=6 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/anti_iden/ppo2_30M_surrogate_0.8 --weight=0.8 --normal=False --surrogate=True --noise_type=anti_iden)&
(export CUDA_VISIBLE_DEVICES=7 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/anti_iden/ppo2_30M_surrogate_0.9 --weight=0.9 --normal=False --surrogate=True --noise_type=anti_iden)&


(export CUDA_VISIBLE_DEVICES=0 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_all/ppo2_30M_noisy_0.1 --weight=0.1 --normal=False --surrogate=False --noise_type=norm_all)&
(export CUDA_VISIBLE_DEVICES=1 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_all/ppo2_30M_noisy_0.2 --weight=0.2 --normal=False --surrogate=False --noise_type=norm_all)&
(export CUDA_VISIBLE_DEVICES=2 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_all/ppo2_30M_noisy_0.3 --weight=0.3 --normal=False --surrogate=False --noise_type=norm_all)&
(export CUDA_VISIBLE_DEVICES=3 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_all/ppo2_30M_noisy_0.4 --weight=0.4 --normal=False --surrogate=False --noise_type=norm_all)&
(export CUDA_VISIBLE_DEVICES=4 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_all/ppo2_30M_noisy_0.6 --weight=0.6 --normal=False --surrogate=False --noise_type=norm_all)&
(export CUDA_VISIBLE_DEVICES=5 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_all/ppo2_30M_noisy_0.7 --weight=0.7 --normal=False --surrogate=False --noise_type=norm_all)&
(export CUDA_VISIBLE_DEVICES=6 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_all/ppo2_30M_noisy_0.8 --weight=0.8 --normal=False --surrogate=False --noise_type=norm_all)&
(export CUDA_VISIBLE_DEVICES=7 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_all/ppo2_30M_noisy_0.9 --weight=0.9 --normal=False --surrogate=False --noise_type=norm_all)&

(export CUDA_VISIBLE_DEVICES=0 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_all/ppo2_30M_surrogate_0.1 --weight=0.1 --normal=False --surrogate=True --noise_type=norm_all)&
(export CUDA_VISIBLE_DEVICES=1 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_all/ppo2_30M_surrogate_0.2 --weight=0.2 --normal=False --surrogate=True --noise_type=norm_all)&
(export CUDA_VISIBLE_DEVICES=2 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_all/ppo2_30M_surrogate_0.3 --weight=0.3 --normal=False --surrogate=True --noise_type=norm_all)&
(export CUDA_VISIBLE_DEVICES=3 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_all/ppo2_30M_surrogate_0.4 --weight=0.4 --normal=False --surrogate=True --noise_type=norm_all)&
(export CUDA_VISIBLE_DEVICES=4 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_all/ppo2_30M_surrogate_0.6 --weight=0.6 --normal=False --surrogate=True --noise_type=norm_all)&
(export CUDA_VISIBLE_DEVICES=5 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_all/ppo2_30M_surrogate_0.7 --weight=0.7 --normal=False --surrogate=True --noise_type=norm_all)&
(export CUDA_VISIBLE_DEVICES=6 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_all/ppo2_30M_surrogate_0.8 --weight=0.8 --normal=False --surrogate=True --noise_type=norm_all)&
(export CUDA_VISIBLE_DEVICES=7 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_all/ppo2_30M_surrogate_0.9 --weight=0.9 --normal=False --surrogate=True --noise_type=norm_all)&


(export CUDA_VISIBLE_DEVICES=0 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_one/ppo2_30M_noisy_0.1 --weight=0.1 --normal=False --surrogate=False --noise_type=norm_one)&
(export CUDA_VISIBLE_DEVICES=1 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_one/ppo2_30M_noisy_0.2 --weight=0.2 --normal=False --surrogate=False --noise_type=norm_one)&
(export CUDA_VISIBLE_DEVICES=2 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_one/ppo2_30M_noisy_0.3 --weight=0.3 --normal=False --surrogate=False --noise_type=norm_one)&
(export CUDA_VISIBLE_DEVICES=3 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_one/ppo2_30M_noisy_0.4 --weight=0.4 --normal=False --surrogate=False --noise_type=norm_one)&
(export CUDA_VISIBLE_DEVICES=4 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_one/ppo2_30M_noisy_0.6 --weight=0.6 --normal=False --surrogate=False --noise_type=norm_one)&
(export CUDA_VISIBLE_DEVICES=5 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_one/ppo2_30M_noisy_0.7 --weight=0.7 --normal=False --surrogate=False --noise_type=norm_one)&
(export CUDA_VISIBLE_DEVICES=6 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_one/ppo2_30M_noisy_0.8 --weight=0.8 --normal=False --surrogate=False --noise_type=norm_one)&
(export CUDA_VISIBLE_DEVICES=7 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_one/ppo2_30M_noisy_0.9 --weight=0.9 --normal=False --surrogate=False --noise_type=norm_one)&

(export CUDA_VISIBLE_DEVICES=0 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_one/ppo2_30M_surrogate_0.1 --weight=0.1 --normal=False --surrogate=True --noise_type=norm_one)&
(export CUDA_VISIBLE_DEVICES=1 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_one/ppo2_30M_surrogate_0.2 --weight=0.2 --normal=False --surrogate=True --noise_type=norm_one)&
(export CUDA_VISIBLE_DEVICES=2 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_one/ppo2_30M_surrogate_0.3 --weight=0.3 --normal=False --surrogate=True --noise_type=norm_one)&
(export CUDA_VISIBLE_DEVICES=3 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_one/ppo2_30M_surrogate_0.4 --weight=0.4 --normal=False --surrogate=True --noise_type=norm_one)&
(export CUDA_VISIBLE_DEVICES=4 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_one/ppo2_30M_surrogate_0.6 --weight=0.6 --normal=False --surrogate=True --noise_type=norm_one)&
(export CUDA_VISIBLE_DEVICES=5 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_one/ppo2_30M_surrogate_0.7 --weight=0.7 --normal=False --surrogate=True --noise_type=norm_one)&
(export CUDA_VISIBLE_DEVICES=6 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_one/ppo2_30M_surrogate_0.8 --weight=0.8 --normal=False --surrogate=True --noise_type=norm_one)&
(export CUDA_VISIBLE_DEVICES=7 && python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=3e7 --save_path=logs-pong/pong/norm_one/ppo2_30M_surrogate_0.9 --weight=0.9 --normal=False --surrogate=True --noise_type=norm_one)&

cd ..
