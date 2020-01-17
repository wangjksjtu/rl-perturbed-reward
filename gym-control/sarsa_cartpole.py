import argparse
import collections
import pandas
import numpy as np
import os
import gym

from keras.layers import Activation, Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import tensorflow as tf

from rl.agents import SARSAAgent
from rl.core import Processor
from rl.policy import BoltzmannQPolicy
from noise_estimator import *
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--error_positive', type=float, default=0.2,
                    help='Error positive rate [default: 0.2]')
parser.add_argument('--error_negative', type=float, default=0.0,
                    help='Error negative rate [default: 0.0]')
parser.add_argument('--log_dir', default='logs',
                    help='Log dir [default: logs]')
parser.add_argument('--reward', default='normal',
                    help='reward choice: normal/noisy/surrogate [default: normal]')
parser.add_argument('--smooth', type=str2bool, default=False,
                    help='Add smoothing to rewards [default: False]')
FLAGS = parser.parse_args()

ERR_P = FLAGS.error_positive
ERR_N = FLAGS.error_negative
REWARD = FLAGS.reward
SMOOTH = FLAGS.smooth

if REWARD == "normal":
    LOG_DIR = os.path.join(FLAGS.log_dir, "sarsa_cartpole")
else:
    LOG_DIR = os.path.join(os.path.join(FLAGS.log_dir, "sarsa_cartpole"), str(ERR_P))
ENV_NAME = 'CartPole-v0'

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
os.system('cp sarsa_cartpole.py %s' % (LOG_DIR)) # bkp of train procedure
print ('cp sarsa_cartpole.py %s' % (LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'setting.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]


def train():
    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    # SARSA does not require a memory.
    policy = BoltzmannQPolicy()
    # processor_noisy = CartpoleSurrogateProcessor(e_= ERR_N, e=ERR_P, surrogate=False)
    # processor_surrogate = CartpoleSurrogateProcessor(e_= ERR_N, e=ERR_P, surrogate=True)
    if not SMOOTH:
        processor_noisy = CartpoleProcessor(e_= ERR_N, e=ERR_P, smooth=False, surrogate=False)
        processor_surrogate = CartpoleProcessor(e_= ERR_N, e=ERR_P, smooth=False, surrogate=True)
    else:
        processor_noisy = CartpoleProcessor(e_= ERR_N, e=ERR_P, smooth=True, surrogate=False)
        processor_surrogate = CartpoleProcessor(e_= ERR_N, e=ERR_P, smooth=True, surrogate=True)        

    if REWARD == "normal":
        sarsa_normal = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=10, 
                                  policy=policy)
        sarsa_normal.compile(Adam(lr=1e-3), metrics=['mae'])
        history_normal = sarsa_normal.fit(env, nb_steps=50000, visualize=False, verbose=2)
        sarsa_normal.save_weights(os.path.join(LOG_DIR, 'sarsa_normal_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
        sarsa_normal.test(env, nb_episodes=10, visualize=False, verbose=2)

        pandas.DataFrame(history_normal.history).to_csv(os.path.join(LOG_DIR, "normal.csv"))


    elif REWARD == "noisy":
        sarsa_noisy = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=10, 
                                 policy=policy, processor=processor_noisy)
        sarsa_noisy.compile(Adam(lr=1e-3), metrics=['mae'])
        history_noisy = sarsa_noisy.fit(env, nb_steps=50000, visualize=False, verbose=2)
        if not SMOOTH:
            sarsa_noisy.save_weights(os.path.join(LOG_DIR, 'sarsa_noisy_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
            pandas.DataFrame(history_noisy.history).to_csv(os.path.join(LOG_DIR, "noisy.csv"))
        else:
            sarsa_noisy.save_weights(os.path.join(LOG_DIR, 'sarsa_noisy_smooth_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
            pandas.DataFrame(history_noisy.history).to_csv(os.path.join(LOG_DIR, "noisy_smooth.csv"))

        sarsa_noisy.test(env, nb_episodes=10, visualize=False)


    elif REWARD == "surrogate":
        sarsa_surrogate = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=10, 
                                     policy=policy, processor=processor_surrogate)
        sarsa_surrogate.compile(Adam(lr=1e-3), metrics=['mae'])
        history_surrogate = sarsa_surrogate.fit(env, nb_steps=50000, visualize=False, verbose=2)
        if not SMOOTH:
            sarsa_surrogate.save_weights(os.path.join(LOG_DIR, 'sarsa_surrogate_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
            pandas.DataFrame(history_surrogate.history).to_csv(os.path.join(LOG_DIR, "surrogate.csv"))

        else:
            sarsa_surrogate.save_weights(os.path.join(LOG_DIR, 'sarsa_surrogate_smooth_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
            pandas.DataFrame(history_surrogate.history).to_csv(os.path.join(LOG_DIR, "surrogate_smooth.csv"))

        sarsa_surrogate.test(env, nb_episodes=10, visualize=False)



if __name__ == "__main__":
    train()