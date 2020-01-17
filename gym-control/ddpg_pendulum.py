import argparse
import pandas
import numpy as np
import os
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
import tensorflow as tf

from rl.agents import DDPGAgent
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess


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
    LOG_DIR = os.path.join(os.path.join(FLAGS.log_dir, "ddpg_pendulum"), NOISE_TYPE)
else:
    LOG_DIR = os.path.join(os.path.join(os.path.join(FLAGS.log_dir, "ddpg_pendulum"), NOISE_TYPE), str(WEIGHT))
ENV_NAME = 'Pendulum-v0'
# gym.undo_logger_setup()

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
os.system('cp ddpg_pendulum.py %s' % (LOG_DIR)) # bkp of train procedure


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

    # Next, we build a very simple model.
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('linear'))
    # print(actor.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    # print(critic.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)

    if REWARD == "normal":
        ddpg_normal = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                                memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                                random_process=random_process, gamma=.99, target_model_update=1e-3)
        ddpg_normal.compile(Adam(lr=.0005, clipnorm=1.), metrics=['mae'])

        # Okay, now it's time to learn something! We visualize the training here for show, but this
        # slows down training quite a lot. You can always safely abort the training prematurely using
        # Ctrl + C.
        history_normal = ddpg_normal.fit(env, nb_steps=150000, visualize=False, verbose=2, nb_max_episode_steps=200)

        # After training is done, we save the final weights.
        ddpg_normal.save_weights(os.path.join(LOG_DIR, 'ddpg_normal_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
        # Finally, evaluate our algorithm for 5 episodes.
        ddpg_normal.test(env, nb_episodes=5, visualize=False, verbose=2, nb_max_episode_steps=200)

        pandas.DataFrame(history_normal.history).to_csv(os.path.join(LOG_DIR, "normal.csv"))

    elif REWARD == "noisy":
        processor_noisy = PendulumSurrogateProcessor(weight=WEIGHT, surrogate=False, noise_type=NOISE_TYPE)
        ddpg_noisy = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                               memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                               random_process=random_process, gamma=.99, target_model_update=1e-3,
                               processor=processor_noisy)
        ddpg_noisy.compile(Adam(lr=.0005, clipnorm=1.), metrics=['mae'])
        history_noisy = ddpg_noisy.fit(env, nb_steps=150000, visualize=False, verbose=2, nb_max_episode_steps=200)
        ddpg_noisy.save_weights(os.path.join(LOG_DIR, 'ddpg_noisy_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
        ddpg_noisy.test(env, nb_episodes=5, visualize=False, verbose=2, nb_max_episode_steps=200)

        pandas.DataFrame(history_noisy.history).to_csv(os.path.join(LOG_DIR, "noisy.csv"))

    elif REWARD == "surrogate":
        processor_surrogate = PendulumSurrogateProcessor(weight=WEIGHT, surrogate=True, noise_type=NOISE_TYPE)
        ddpg_surrogate = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                                   memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                                   random_process=random_process, gamma=.99, target_model_update=1e-3,
                                   processor=processor_surrogate)
        ddpg_surrogate.compile(Adam(lr=.0005, clipnorm=1.), metrics=['mae'])
        history_surrogate = ddpg_surrogate.fit(env, nb_steps=150000, visualize=False, verbose=2, nb_max_episode_steps=200)

        ddpg_surrogate.save_weights(os.path.join(LOG_DIR, 'ddpg_surrogate_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
        ddpg_surrogate.test(env, nb_episodes=5, visualize=False, verbose=2, nb_max_episode_steps=200)

        pandas.DataFrame(history_surrogate.history).to_csv(os.path.join(LOG_DIR, "surrogate.csv"))

    else:
        raise NotImplementedError

if __name__ == "__main__":
    train()
