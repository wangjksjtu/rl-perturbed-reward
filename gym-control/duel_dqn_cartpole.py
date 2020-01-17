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

from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.memory import SequentialMemory
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
                    help='Reward choice: normal/noisy/surrogate [default: normal]')
parser.add_argument('--smooth', type=str2bool, default=False,
                    help='Add smoothing to rewards [default: False]')
FLAGS = parser.parse_args()

ERR_P = FLAGS.error_positive
ERR_N = FLAGS.error_negative
REWARD = FLAGS.reward
SMOOTH = FLAGS.smooth

if REWARD == "normal":
    LOG_DIR = os.path.join(FLAGS.log_dir, "duel_dqn_cartpole")
else:
    LOG_DIR = os.path.join(os.path.join(FLAGS.log_dir, "duel_dqn_cartpole"), str(ERR_P))
ENV_NAME = 'CartPole-v0'

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
os.system('cp duel_dqn_cartpole.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'setting.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


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
    model.summary()

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    if REWARD == "normal":
        dqn_normal = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                              enable_dueling_network=True, dueling_type='avg',
                              target_model_update=1e-2, policy=policy)
        dqn_normal.compile(Adam(lr=1e-3), metrics=['mae'])
        history_normal = dqn_normal.fit(env, nb_steps=10000, visualize=False, verbose=2)
        dqn_normal.save_weights(os.path.join(LOG_DIR, 'duel_dqn_normal_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
        dqn_normal.test(env, nb_episodes=10, visualize=False, verbose=2)
        
        pandas.DataFrame(history_normal.history).to_csv(os.path.join(LOG_DIR, "normal.csv"))

    elif REWARD == "noisy":
        if not SMOOTH:
            processor_noisy = CartpoleProcessor(e_=ERR_N, e=ERR_P, smooth=False, surrogate=False)
        else:
            processor_noisy = CartpoleProcessor(e_=ERR_N, e=ERR_P, smooth=True, surrogate=False)
        
        # processor_noisy = CartpoleSurrogateProcessor(e_=ERR_N, e=ERR_P, surrogate=False)
        dqn_noisy = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                             enable_dueling_network=True, dueling_type='avg',
                             target_model_update=1e-2, policy=policy, processor=processor_noisy)
        dqn_noisy.compile(Adam(lr=1e-3), metrics=['mae'])
        history_noisy = dqn_noisy.fit(env, nb_steps=10000, visualize=False, verbose=2)
        if not SMOOTH:
            dqn_noisy.save_weights(os.path.join(LOG_DIR, 'duel_dqn_noisy_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
            pandas.DataFrame(history_noisy.history).to_csv(os.path.join(LOG_DIR, "noisy.csv"))
        else:
            dqn_noisy.save_weights(os.path.join(LOG_DIR, 'duel_dqn_noisy_smooth_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
            pandas.DataFrame(history_noisy.history).to_csv(os.path.join(LOG_DIR, "noisy_smooth.csv"))

        dqn_noisy.test(env, nb_episodes=10, visualize=False, verbose=2)
        

    elif REWARD == "surrogate":
        if not SMOOTH:
            processor_surrogate = CartpoleProcessor(e_=ERR_N, e=ERR_P, smooth=False, surrogate=True)
        else:
            processor_surrogate = CartpoleProcessor(e_=ERR_N, e=ERR_P, smooth=True, surrogate=True)

        # processor_surrogate = CartpoleSurrogateProcessor(e_=ERR_N, e=ERR_P, surrogate=True)
        dqn_surrogate = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                                 enable_dueling_network=True, dueling_type='avg',
                                 target_model_update=1e-2, policy=policy, processor=processor_surrogate)
        dqn_surrogate.compile(Adam(lr=1e-3), metrics=['mae'])    
        history_surrogate = dqn_surrogate.fit(env, nb_steps=10000, visualize=False, verbose=2)
        if not SMOOTH:
            dqn_surrogate.save_weights(os.path.join(LOG_DIR, 'duel_dqn_surrogate_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
            pandas.DataFrame(history_surrogate.history).to_csv(os.path.join(LOG_DIR, "surrogate.csv"))
        else:
            dqn_surrogate.save_weights(os.path.join(LOG_DIR, 'duel_dqn_surrogate_smooth_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
            pandas.DataFrame(history_surrogate.history).to_csv(os.path.join(LOG_DIR, "surrogate_smooth.csv"))

        dqn_surrogate.test(env, nb_episodes=10, visualize=False, verbose=2)

    else:
        raise NotImplementedError

if __name__ == "__main__":
    train()