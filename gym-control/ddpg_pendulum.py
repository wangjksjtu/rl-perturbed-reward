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
gym.undo_logger_setup()

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
os.system('cp ddpg_pendulum.py %s' % (LOG_DIR)) # bkp of train procedure
if REWARD == "noisy":
    LOG_FOUT = open(os.path.join(LOG_DIR, 'noisy_reward'), 'w')
elif REWARD == "surrogate":
    LOG_FOUT = open(os.path.join(LOG_DIR, 'surrogate_reward'), 'w')
else:
    pass


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def initialize_cmat(noise_type, M, weight):
    cmat = None
    flag = True
    cnt = 0
    while flag:
        if noise_type == "norm_all":
            init_norm = np.random.rand(M, M) # reward: 0 ~ -16
            cmat = init_norm / init_norm.sum(axis=1, keepdims=1) * weight + \
                   (1 - weight) * np.identity(M)

        elif noise_type == "norm_one":
            i_mat = np.identity(M)
            map(np.random.shuffle, i_mat)
            cmat = i_mat * weight + (1 - weight) * np.identity(M)

        elif noise_type == "anti_iden":
            # if weight == 0.5: raise ValueError
            cmat = np.identity(M)[::-1] * weight + \
                   (1 - weight) * np.identity(M)
            if weight == 0.5: break
        else:
            # if weight == 0.5: raise ValueError
            i1_mat = np.zeros((M, M)); i1_mat[0:M/2, -1] = 1; i1_mat[M/2:, 0] = 1
            i2_mat = np.zeros((M, M)); i2_mat[0:int(np.ceil(M/2.0)), -1] = 1; i2_mat[int(np.ceil(M/2.0)):, 0] = 1
            i_mat = (i1_mat + i2_mat) / 2.0
            cmat = i_mat * weight + (1 - weight) * np.identity(M)
            if weight == 0.5: break
        if is_invertible(cmat):
            flag = False
        cnt += 1

    return cmat, cnt


class SurrogateRewardProcessor(Processor):
    """
    Learning from surrogate reward
    following paper "Learning from noisy labels"
    """
    def __init__(self, weight=0.6, surrogate=False, noise_type="norm_all"):
        M = 17
        self.weight = weight
        self.surrogate = surrogate
        self.cmat, _ = initialize_cmat(noise_type, M, self.weight)
        # assert (is_invertible(self.cmat))

        self.cummat = np.cumsum(self.cmat, axis=1)
        # print self.cummat
        self.mmat = np.expand_dims(np.asarray(range(0, -1* M, -1)), axis=1)
        print self.cmat.T.shape, self.mmat.shape
        self.phi = self.cmat.T.dot(self.mmat)
        print self.phi.shape
        self.r_sum = 0
        self.r_counter = 0

    def noisy_reward(self, reward):
        prob_list = list(self.cummat[abs(reward), :])
        # print prob_list
        n = np.random.random()
        # print n
        prob_list.append(n)
        # print sorted(prob_list)
        j = sorted(prob_list).index(n)
        # print j
        reward = -1 * j
        # print reward
        return reward

    def process_reward(self, reward):
        self.r_sum += reward
        self.r_counter += 1
        if self.r_counter == 200:
            log_string(str(self.r_sum / float(self.r_counter)))
            self.r_counter = 0
            self.r_sum = 0

        reward = int(np.ceil(reward))

        r = self.noisy_reward(reward)

        if self.surrogate:
            return self.phi[int(-r), 0]
        return r

    '''
    def process_observation(self, observation):
        self.o_sum += observation[0:2]
        self.o_counter += 1
        if self.o_counter == 200:
            print "sin(x): ", self.o_sum[1] / self.o_counter, "cos(x): ", self.o_sum[0] / self.o_counter
            self.o_sum = 0
            self.o_counter = 0

        return observation
    '''

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
        history_normal = ddpg_normal.fit(env, nb_steps=150000, visualize=False, verbose=0, nb_max_episode_steps=200)

        # After training is done, we save the final weights.
        ddpg_normal.save_weights(os.path.join(LOG_DIR, 'ddpg_normal_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
        # Finally, evaluate our algorithm for 5 episodes.
        ddpg_normal.test(env, nb_episodes=5, visualize=False, verbose=0, nb_max_episode_steps=200)

        pandas.DataFrame(history_normal.history).to_csv(os.path.join(LOG_DIR, "normal.csv"))

    elif REWARD == "noisy":
        processor_noisy = SurrogateRewardProcessor(weight=WEIGHT, surrogate=False, noise_type=NOISE_TYPE)
        ddpg_noisy = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                               memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                               random_process=random_process, gamma=.99, target_model_update=1e-3,
                               processor=processor_noisy)
        ddpg_noisy.compile(Adam(lr=.0005, clipnorm=1.), metrics=['mae'])
        history_noisy = ddpg_noisy.fit(env, nb_steps=150000, visualize=False, verbose=0, nb_max_episode_steps=200)
        ddpg_noisy.save_weights(os.path.join(LOG_DIR, 'ddpg_noisy_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
        ddpg_noisy.test(env, nb_episodes=5, visualize=False, verbose=0, nb_max_episode_steps=200)

        pandas.DataFrame(history_noisy.history).to_csv(os.path.join(LOG_DIR, "noisy.csv"))

    elif REWARD == "surrogate":
        processor_surrogate = SurrogateRewardProcessor(weight=WEIGHT, surrogate=True, noise_type=NOISE_TYPE)
        ddpg_surrogate = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                                   memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                                   random_process=random_process, gamma=.99, target_model_update=1e-3,
                                   processor=processor_surrogate)
        ddpg_surrogate.compile(Adam(lr=.0005, clipnorm=1.), metrics=['mae'])
        history_surrogate = ddpg_surrogate.fit(env, nb_steps=150000, visualize=False, verbose=0, nb_max_episode_steps=200)

        ddpg_surrogate.save_weights(os.path.join(LOG_DIR, 'ddpg_surrogate_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
        ddpg_surrogate.test(env, nb_episodes=5, visualize=False, verbose=0, nb_max_episode_steps=200)

        pandas.DataFrame(history_surrogate.history).to_csv(os.path.join(LOG_DIR, "surrogate.csv"))

    else:
        raise NotImplementedError

if __name__ == "__main__":
    train()
