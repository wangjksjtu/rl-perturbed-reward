for i in $(seq 1 3);
do
for log_dir in qlearn/$i
do
(python qlearn_cartpole.py --log_dir $log_dir)&

(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.1 --reward noisy)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.2 --reward noisy)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.3 --reward noisy)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.4 --reward noisy)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.9 --reward noisy)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.6 --reward noisy)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.7 --reward noisy)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.8 --reward noisy)&

(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.1 --reward noisy --smooth True)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.2 --reward noisy --smooth True)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.3 --reward noisy --smooth True)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.4 --reward noisy --smooth True)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.9 --reward noisy --smooth True)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.6 --reward noisy --smooth True)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.7 --reward noisy --smooth True)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.8 --reward noisy --smooth True)&

(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.1 --reward surrogate)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.2 --reward surrogate)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.3 --reward surrogate)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.4 --reward surrogate)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.9 --reward surrogate)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.6 --reward surrogate)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.7 --reward surrogate)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.8 --reward surrogate)&

(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.1 --reward surrogate --smooth True)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.2 --reward surrogate --smooth True)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.3 --reward surrogate --smooth True)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.4 --reward surrogate --smooth True)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.9 --reward surrogate --smooth True)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.6 --reward surrogate --smooth True)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.7 --reward surrogate --smooth True)&
(python qlearn_cartpole.py --log_dir $log_dir --error_positive 0.8 --reward surrogate --smooth True)&
done
done
