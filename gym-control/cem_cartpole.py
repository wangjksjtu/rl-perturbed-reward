import argparse
import collections
import pandas
import numpy as np
import os
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import tensorflow as tf

from rl.agents.cem import CEMAgent
from rl.memory import EpisodeParameterMemory
from noise_estimator import CartpoleProcessor, CartpoleSurrogateProcessor
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
    LOG_DIR = os.path.join(FLAGS.log_dir, "cem_cartpole")
else:
    LOG_DIR = os.path.join(os.path.join(FLAGS.log_dir, "cem_cartpole"), str(ERR_P))
ENV_NAME = 'CartPole-v0'

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
os.system('cp cem_cartpole.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'setting.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

def train():
    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)

    nb_actions = env.action_space.n
    obs_dim = env.observation_space.shape[0]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    # Option 1 : Simple model
    # model = Sequential()
    # model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    # model.add(Dense(nb_actions))
    # model.add(Activation('softmax'))

    # Option 2: deep network
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('softmax'))

    model.summary()

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = EpisodeParameterMemory(limit=1000, window_length=1)

    if REWARD == "normal":
        cem = CEMAgent(model=model, nb_actions=nb_actions, memory=memory,
                       batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)
        cem.compile()
        history_normal = cem.fit(env, nb_steps=100000, visualize=False, verbose=2)
        cem.save_weights(os.path.join(LOG_DIR, 'cem_normal_{}_params.h5f'.format(ENV_NAME)), overwrite=True)
        cem.test(env, nb_episodes=5, visualize=False)

        pandas.DataFrame(history_normal.history).to_csv(os.path.join(LOG_DIR, "normal.csv"))

    elif REWARD == "noisy":
        if not SMOOTH:
            processor_noisy = CartpoleProcessor(e_=ERR_N, e=ERR_P, smooth=False, surrogate=False)
        else:
            processor_noisy = CartpoleProcessor(e_=ERR_N, e=ERR_P, smooth=True, surrogate=False)

        # processor_surrogate = CartpoleSurrogateProcessor(e_=ERR_N, e=ERR_P, surrogate=False)
        cem = CEMAgent(model=model, nb_actions=nb_actions, memory=memory,
                       batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05,
                       processor=processor_noisy)
        cem.compile()
        history_noisy = cem.fit(env, nb_steps=100000, visualize=False, verbose=2)
        if not SMOOTH:
            cem.save_weights(os.path.join(LOG_DIR, 'cem_noisy_{}_params.h5f'.format(ENV_NAME)), overwrite=True)
            pandas.DataFrame(history_noisy.history).to_csv(os.path.join(LOG_DIR, "noisy.csv"))

        else:
            cem.save_weights(os.path.join(LOG_DIR, 'cem_noisy_smooth_{}_params.h5f'.format(ENV_NAME)), overwrite=True)
            pandas.DataFrame(history_noisy.history).to_csv(os.path.join(LOG_DIR, "noisy_smooth.csv"))

        cem.test(env, nb_episodes=5, visualize=False)

    elif REWARD == "surrogate":
        if not SMOOTH:
            processor_surrogate = CartpoleProcessor(e_=ERR_N, e=ERR_P, smooth=False, surrogate=True)
        else:
            processor_surrogate = CartpoleProcessor(e_=ERR_N, e=ERR_P, smooth=True, surrogate=True)

        # processor_surrogate = CartpoleSurrogateProcessor(e_=ERR_N, e=ERR_P, surrogate=True)
        cem = CEMAgent(model=model, nb_actions=nb_actions, memory=memory,
                       batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05,
                       processor=processor_surrogate)
        cem.compile()
        history_surrogate = cem.fit(env, nb_steps=100000, visualize=False, verbose=2)
        if not SMOOTH:
            cem.save_weights(os.path.join(LOG_DIR, 'cem_surrogate_{}_params.h5f'.format(ENV_NAME)), overwrite=True)
            pandas.DataFrame(history_surrogate.history).to_csv(os.path.join(LOG_DIR, "surrogate.csv"))
        else:
            cem.save_weights(os.path.join(LOG_DIR, 'cem_surrogate_smooth_{}_params.h5f'.format(ENV_NAME)), overwrite=True)
            pandas.DataFrame(history_surrogate.history).to_csv(os.path.join(LOG_DIR, "surrogate_smooth.csv"))

        cem.test(env, nb_episodes=5, visualize=False)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    train()
