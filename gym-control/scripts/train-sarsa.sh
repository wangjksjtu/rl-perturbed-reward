for log_dir in logs_01 logs_02 logs_03
do
(export CUDA_VISIBLE_DEVICES=0 && python sarsa_cartpole.py --log_dir $log_dir)&

(export CUDA_VISIBLE_DEVICES=0 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.1 --reward noisy)&
(export CUDA_VISIBLE_DEVICES=1 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.2 --reward noisy)&
(export CUDA_VISIBLE_DEVICES=2 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.3 --reward noisy)&
(export CUDA_VISIBLE_DEVICES=3 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.4 --reward noisy)&
(export CUDA_VISIBLE_DEVICES=4 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.9 --reward noisy)&
(export CUDA_VISIBLE_DEVICES=5 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.6 --reward noisy)&
(export CUDA_VISIBLE_DEVICES=6 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.7 --reward noisy)&
(export CUDA_VISIBLE_DEVICES=7 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.8 --reward noisy)&

(export CUDA_VISIBLE_DEVICES=0 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.1 --reward noisy --smooth True)&
(export CUDA_VISIBLE_DEVICES=1 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.2 --reward noisy --smooth True)&
(export CUDA_VISIBLE_DEVICES=2 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.3 --reward noisy --smooth True)&
(export CUDA_VISIBLE_DEVICES=3 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.4 --reward noisy --smooth True)&
(export CUDA_VISIBLE_DEVICES=4 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.9 --reward noisy --smooth True)&
(export CUDA_VISIBLE_DEVICES=5 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.6 --reward noisy --smooth True)&
(export CUDA_VISIBLE_DEVICES=6 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.7 --reward noisy --smooth True)&
(export CUDA_VISIBLE_DEVICES=7 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.8 --reward noisy --smooth True)&

(export CUDA_VISIBLE_DEVICES=0 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.1 --reward surrogate)&
(export CUDA_VISIBLE_DEVICES=1 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.2 --reward surrogate)&
(export CUDA_VISIBLE_DEVICES=2 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.3 --reward surrogate)&
(export CUDA_VISIBLE_DEVICES=3 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.4 --reward surrogate)&
(export CUDA_VISIBLE_DEVICES=4 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.9 --reward surrogate)&
(export CUDA_VISIBLE_DEVICES=5 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.6 --reward surrogate)&
(export CUDA_VISIBLE_DEVICES=6 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.7 --reward surrogate)&
(export CUDA_VISIBLE_DEVICES=7 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.8 --reward surrogate)&

(export CUDA_VISIBLE_DEVICES=0 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.1 --reward surrogate --smooth True)&
(export CUDA_VISIBLE_DEVICES=1 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.2 --reward surrogate --smooth True)&
(export CUDA_VISIBLE_DEVICES=2 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.3 --reward surrogate --smooth True)&
(export CUDA_VISIBLE_DEVICES=3 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.4 --reward surrogate --smooth True)&
(export CUDA_VISIBLE_DEVICES=4 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.9 --reward surrogate --smooth True)&
(export CUDA_VISIBLE_DEVICES=5 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.6 --reward surrogate --smooth True)&
(export CUDA_VISIBLE_DEVICES=6 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.7 --reward surrogate --smooth True)&
(export CUDA_VISIBLE_DEVICES=7 && python sarsa_cartpole.py --log_dir $log_dir --error_positive 0.8 --reward surrogate --smooth True)&
done
