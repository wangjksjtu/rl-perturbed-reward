import argparse
import os

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default='baselines/logs',
                    help='The path of log directory [default: baselines/logs')
parser.add_argument('--all', type=str2bool, default=False,
                    help='Plot all the curves (diff errs) [default: False]')
parser.add_argument('--weight', type=float, default=0.2,
                    help='Weight of noise [default: 0.2]')
parser.add_argument('--noise_type', type=str, default='anti_iden',
                    help='Type of additional noise [default: anti_iden]')
parser.add_argument('--save_dir', type=str, default='../results',
                    help='Path of root directory to save plots [default: save_dir]')
parser.add_argument('--env_name', type=str, default='Pong',
                    help='Name of Atari game')
parser.add_argument('--num_timesteps', type=int, default=5e7,
                    help='Number of timesteps')

FLAGS = parser.parse_args()

LOG_DIR = FLAGS.log_dir
ALL = FLAGS.all
WEIGHT = FLAGS.weight
NOISE_TYPE = FLAGS.noise_type
SAVE_DIR = FLAGS.save_dir
ENV = FLAGS.env_name
NUM_TIMESTEPS = FLAGS.num_timesteps

assert (os.path.exists(LOG_DIR))
assert (NOISE_TYPE in ['norm_one', 'norm_all', 'anti_iden'])

SAVE_DIR = os.path.join(SAVE_DIR, ENV)
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def visualize():
    if ALL:
        weights_list = [0.1, 0.2, 0.3, 0.4, 
                        0.6, 0.7, 0.8, 0.9]
        if NOISE_TYPE != "anti_iden":
            weights_list.append(0.5)
    else:
        weights_list = [WEIGHT]
    
    for weight in weights_list:
        print ("python -m baselines.results_compare --log_dir %s --task_name %s \
                   --weight %s --noise_type %s --num_timesteps %s --save_dir %s" % \
                  (LOG_DIR, ENV, str(weight), NOISE_TYPE, str(NUM_TIMESTEPS), SAVE_DIR))
        os.system("python -m baselines.results_compare --log_dir %s --task_name %s \
                   --weight %s --noise_type %s --num_timesteps %s --save_dir %s" % \
                  (LOG_DIR, ENV, str(weight), NOISE_TYPE, str(NUM_TIMESTEPS), SAVE_DIR))
        print (LOG_DIR, ENV, str(weight), NOISE_TYPE, str(NUM_TIMESTEPS), SAVE_DIR)
    #os.system("cd ..")

if __name__ == "__main__":
    visualize()