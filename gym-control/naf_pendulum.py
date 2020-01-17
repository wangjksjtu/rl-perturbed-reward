import argparse
import collections
import pandas
import numpy as np
import os
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
import tensorflow as tf

from rl.agents import NAFAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor
from noise_estimator import *


parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='logs',
                    help='Log dir [default: logs]')
parser.add_argument('--reward', default='normal',
                    help='reward choice: normal/noisy/surrogate [default: normal]')
parser.add_argument('--weight', type=float, default=0.6,
                    help='Weight of random confusion matrix [default: 0.6]')
parser.add_argument('--noise_type', type=str, default='norm_all',
                    help='Type of noise added: norm_all/norm_one/anti_iden/max_one [default: norm_all]')
FLAGS = parser.parse_args()


REWARD = FLAGS.reward
WEIGHT = FLAGS.weight
NOISE_TYPE = FLAGS.noise_type

assert (NOISE_TYPE in ["norm_all", "norm_one", "anti_iden", "max_one"])

if REWARD == "normal":
    LOG_DIR = os.path.join(os.path.join(FLAGS.log_dir, "naf_pendulum"), "normal")
else:
    LOG_DIR = os.path.join(os.path.join(os.path.join(FLAGS.log_dir, "naf_pendulum"), NOISE_TYPE), str(WEIGHT))
ENV_NAME = 'Pendulum-v0'
# gym.undo_logger_setup()

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
os.system('cp naf_pendulum.py %s' % (LOG_DIR)) # bkp of train procedure


def train():
    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)
    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    # Build all necessary models: V, mu, and L networks.
    V_model = Sequential()
    V_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    V_model.add(Dense(16))
    V_model.add(Activation('relu'))
    V_model.add(Dense(16))
    V_model.add(Activation('relu'))
    V_model.add(Dense(16))
    V_model.add(Activation('relu'))
    V_model.add(Dense(1))
    V_model.add(Activation('linear'))
    V_model.summary()

    mu_model = Sequential()
    mu_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    mu_model.add(Dense(16))
    mu_model.add(Activation('relu'))
    mu_model.add(Dense(16))
    mu_model.add(Activation('relu'))
    mu_model.add(Dense(16))
    mu_model.add(Activation('relu'))
    mu_model.add(Dense(nb_actions))
    mu_model.add(Activation('linear'))
    mu_model.summary()

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    x = Concatenate()([action_input, Flatten()(observation_input)])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(((nb_actions * nb_actions + nb_actions) // 2))(x)
    x = Activation('linear')(x)
    L_model = Model(inputs=[action_input, observation_input], outputs=x)
    L_model.summary()

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)

    if REWARD == "normal":
        processor = NAFPendulumProcessor()
        naf_normal = NAFAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                         memory=memory, nb_steps_warmup=100, random_process=random_process,
                         gamma=.99, target_model_update=1e-3, processor=processor)
        naf_normal.compile(Adam(lr=.00025, clipnorm=1.), metrics=['mae'])
        history_normal = naf_normal.fit(env, nb_steps=150000, visualize=False, verbose=2, nb_max_episode_steps=200)
        naf_normal.save_weights(os.path.join(LOG_DIR, 'naf_normal_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
        naf_normal.test(env, nb_episodes=10, visualize=False, nb_max_episode_steps=200)

        pandas.DataFrame(history_normal.history).to_csv(os.path.join(LOG_DIR, "normal.csv"))

    elif REWARD == "noisy":
        # processor_noisy = PendulumSurrogateProcessor(weight=WEIGHT, surrogate=False, noise_type=NOISE_TYPE)
        processor_noisy = PendulumProcessor(weight=WEIGHT, surrogate=False, noise_type=NOISE_TYPE)
        naf_noisy = NAFAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                             memory=memory, nb_steps_warmup=100, random_process=random_process,
                             gamma=.99, target_model_update=1e-3, processor=processor_noisy)
        naf_noisy.compile(Adam(lr=.00025, clipnorm=1.), metrics=['mae'])
        history_noisy = naf_noisy.fit(env, nb_steps=150000, visualize=False, verbose=2, nb_max_episode_steps=200)
        naf_noisy.save_weights(os.path.join(LOG_DIR, 'naf_noisy_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
        naf_noisy.test(env, nb_episodes=10, visualize=False, nb_max_episode_steps=200)

        pandas.DataFrame(history_noisy.history).to_csv(os.path.join(LOG_DIR, "noisy.csv"))

    elif REWARD == "surrogate":
        # processor_surrogate = PendulumSurrogateProcessor(weight=WEIGHT, surrogate=True, noise_type=NOISE_TYPE)
        processor_surrogate = PendulumProcessor(weight=WEIGHT, surrogate=True, noise_type=NOISE_TYPE)
        naf_surrogate = NAFAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                                 memory=memory, nb_steps_warmup=100, random_process=random_process,
                                 gamma=.99, target_model_update=1e-3, processor=processor_surrogate)
        naf_surrogate.compile(Adam(lr=.00025, clipnorm=1.), metrics=['mae'])
        history_surrogate = naf_surrogate.fit(env, nb_steps=150000, visualize=False, verbose=2, nb_max_episode_steps=200)
        naf_surrogate.save_weights(os.path.join(LOG_DIR, 'naf_surrogate_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
        naf_surrogate.test(env, nb_episodes=10, visualize=False, nb_max_episode_steps=200)

        pandas.DataFrame(history_surrogate.history).to_csv(os.path.join(LOG_DIR, "surrogate.csv"))

    else:
        raise NotImplementedError


if __name__ == "__main__":
    train()
