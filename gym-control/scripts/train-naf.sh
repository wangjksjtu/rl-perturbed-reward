for log_dir in naf/1
do
(export CUDA_VISIBLE_DEVICES=5 && python naf_pendulum2.py --weight 0.1 --reward noisy --noise_type norm_all --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=5 && python naf_pendulum2.py --weight 0.2 --reward noisy --noise_type norm_all --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=5 && python naf_pendulum2.py --weight 0.3 --reward noisy --noise_type norm_all --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=5 && python naf_pendulum2.py --weight 0.4 --reward noisy --noise_type norm_all --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=5 && python naf_pendulum2.py --weight 0.9 --reward noisy --noise_type norm_all --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=5 && python naf_pendulum2.py --weight 0.6 --reward noisy --noise_type norm_all --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=5 && python naf_pendulum2.py --weight 0.7 --reward noisy --noise_type norm_all --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=5 && python naf_pendulum2.py --weight 0.8 --reward noisy --noise_type norm_all --log_dir $log_dir)&

(export CUDA_VISIBLE_DEVICES=5 && python naf_pendulum2.py --weight 0.1 --reward surrogate --noise_type norm_all --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=5 && python naf_pendulum2.py --weight 0.2 --reward surrogate --noise_type norm_all --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=5 && python naf_pendulum2.py --weight 0.3 --reward surrogate --noise_type norm_all --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=5 && python naf_pendulum2.py --weight 0.4 --reward surrogate --noise_type norm_all --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=5 && python naf_pendulum2.py --weight 0.9 --reward surrogate --noise_type norm_all --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=5 && python naf_pendulum2.py --weight 0.6 --reward surrogate --noise_type norm_all --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=5 && python naf_pendulum2.py --weight 0.7 --reward surrogate --noise_type norm_all --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=5 && python naf_pendulum2.py --weight 0.8 --reward surrogate --noise_type norm_all --log_dir $log_dir)&

(export CUDA_VISIBLE_DEVICES=5 && python naf_pendulum2.py --weight 0.1 --reward noisy --noise_type norm_one --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=5 && python naf_pendulum2.py --weight 0.2 --reward noisy --noise_type norm_one --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=5 && python naf_pendulum2.py --weight 0.3 --reward noisy --noise_type norm_one --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=5 && python naf_pendulum2.py --weight 0.4 --reward noisy --noise_type norm_one --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=5 && python naf_pendulum2.py --weight 0.9 --reward noisy --noise_type norm_one --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=5 && python naf_pendulum2.py --weight 0.6 --reward noisy --noise_type norm_one --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=5 && python naf_pendulum2.py --weight 0.7 --reward noisy --noise_type norm_one --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=5 && python naf_pendulum2.py --weight 0.8 --reward noisy --noise_type norm_one --log_dir $log_dir)&

(export CUDA_VISIBLE_DEVICES=6 && python naf_pendulum2.py --weight 0.1 --reward surrogate --noise_type norm_one --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=6 && python naf_pendulum2.py --weight 0.2 --reward surrogate --noise_type norm_one --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=6 && python naf_pendulum2.py --weight 0.3 --reward surrogate --noise_type norm_one --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=6 && python naf_pendulum2.py --weight 0.4 --reward surrogate --noise_type norm_one --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=6 && python naf_pendulum2.py --weight 0.9 --reward surrogate --noise_type norm_one --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=6 && python naf_pendulum2.py --weight 0.6 --reward surrogate --noise_type norm_one --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=6 && python naf_pendulum2.py --weight 0.7 --reward surrogate --noise_type norm_one --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=6 && python naf_pendulum2.py --weight 0.8 --reward surrogate --noise_type norm_one --log_dir $log_dir)&

(export CUDA_VISIBLE_DEVICES=6 && python naf_pendulum2.py --weight 0.1 --reward noisy --noise_type anti_iden --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=6 && python naf_pendulum2.py --weight 0.2 --reward noisy --noise_type anti_iden --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=6 && python naf_pendulum2.py --weight 0.3 --reward noisy --noise_type anti_iden --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=6 && python naf_pendulum2.py --weight 0.4 --reward noisy --noise_type anti_iden --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=6 && python naf_pendulum2.py --weight 0.9 --reward noisy --noise_type anti_iden --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=6 && python naf_pendulum2.py --weight 0.6 --reward noisy --noise_type anti_iden --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=6 && python naf_pendulum2.py --weight 0.7 --reward noisy --noise_type anti_iden --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=6 && python naf_pendulum2.py --weight 0.8 --reward noisy --noise_type anti_iden --log_dir $log_dir)&

(export CUDA_VISIBLE_DEVICES=6 && python naf_pendulum2.py --weight 0.1 --reward surrogate --noise_type anti_iden --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=6 && python naf_pendulum2.py --weight 0.2 --reward surrogate --noise_type anti_iden --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=6 && python naf_pendulum2.py --weight 0.3 --reward surrogate --noise_type anti_iden --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=6 && python naf_pendulum2.py --weight 0.4 --reward surrogate --noise_type anti_iden --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=6 && python naf_pendulum2.py --weight 0.9 --reward surrogate --noise_type anti_iden --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=6 && python naf_pendulum2.py --weight 0.6 --reward surrogate --noise_type anti_iden --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=6 && python naf_pendulum2.py --weight 0.7 --reward surrogate --noise_type anti_iden --log_dir $log_dir)&
(export CUDA_VISIBLE_DEVICES=6 && python naf_pendulum2.py --weight 0.8 --reward surrogate --noise_type anti_iden --log_dir $log_dir)&

(export CUDA_VISIBLE_DEVICES=6 && python naf_pendulum2.py --log_dir $log_dir)&
done
