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
    LOG_DIR = os.path.join(FLAGS.log_dir, "dqn_cartpole")
else:
    LOG_DIR = os.path.join(os.path.join(FLAGS.log_dir, "dqn_cartpole"), str(ERR_P))
ENV_NAME = 'CartPole-v0'

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
os.system('cp dqn_cartpole.py %s' % (LOG_DIR)) # bkp of train procedure
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

class PostProcessor(Processor):
    """
    Learning from surrogate reward
    following paper "Learning from noisy labels"
    """
    def __init__(self, e_=0.1, e=0.3, smooth=False, surrogate=False, epsilon=1e-6):
        assert (np.abs(e_ + e - 1) > epsilon)
        self.smooth = smooth
        self.surrogate = surrogate
        self.r_smooth = {}
        self.r_sets = {}
        self.e_ = e_
        self.e = e

        self.r1 = -1
        self.r2 = 1
        self.counter = 0
        self.C = np.identity(2)
        self.epsilon = epsilon
        if self.e > 0.5:
            self.reverse = True
        else: self.reverse = False

    def noisy_reward(self, reward):
        n = np.random.random()
        # print (n, self.e, reward)
        if np.abs(reward - self.r1) < self.epsilon:
            if (n < self.e_):
                return self.r2
        else:
            if (n < self.e):
                return self.r1

        return reward
    
    def smooth_reward(self, state, action, reward):
        if self.r_smooth.has_key((state, action)):
            if len(self.r_smooth[(state, action)]) >= 100:
                self.r_smooth[(state, action)].pop(0)
                self.r_smooth[(state, action)].append(reward)
                return sum(self.r_smooth[(state, action)]) / float(len(self.r_smooth[(state, action)]))
            else:
                self.r_smooth[(state, action)].append(reward)
        else:
            self.r_smooth[(state, action)] = [reward]

        return reward

    def process_reward(self, reward):
        if not self.surrogate:
            return reward

        self.estimate_C()
        self.est_e_ = self.C[0, 1] 
        self.est_e = self.C[1, 0]

        if np.abs(reward - self.r1) < self.epsilon:
            r_surrogate = ((1 - self.est_e) * self.r1 - self.est_e_ * self.r2) / (1 - self.est_e_ - self.est_e)
        else:
            r_surrogate = ((1 - self.est_e_) * self.r2 - self.est_e * self.r1) / (1 - self.est_e_ - self.est_e)

        return r_surrogate
    
    def estimate_C(self):
        if self.counter >= 100 and self.counter % 50 == 0:
            e_ = 0; e = 0

            self.count1 = 0
            self.count2 = 0
            for k in self.r_sets.keys():
                freq_count = collections.Counter(self.r_sets[k])
                
                if self.reverse:
                    truth, count = freq_count.most_common()[-1]
                else: 
                    truth, count = freq_count.most_common()[0]
                
                if truth == self.r1:
                    self.count1 += len(self.r_sets[k])
                else:
                    self.count2 += len(self.r_sets[k])

            # print (self.count1, self.count2)
            for k in self.r_sets.keys():
                freq_count = collections.Counter(self.r_sets[k])
                
                if self.reverse:
                    truth, count = freq_count.most_common()[-1]
                else:
                    truth, count = freq_count.most_common()[0]

                prob_correct = float(count) / len(self.r_sets[k])
                # print (prob_correct)
                if truth == self.r1:
                    if self.count1 > 800:
                        prob_k = float(len(self.r_sets[k])) / self.count1
                        e_ += prob_k * (1 - prob_correct)
                    else: e_ = 0.0
                else:
                    prob_k = float(len(self.r_sets[k])) / self.count2
                    e += prob_k * (1 - prob_correct)
                
            # print prob
            log_string(str(e_) + " " + str(e))
            self.C = np.array([[1-e_, e_], [e, 1-e]])
            
            # if self.counter >= 10000:
            #     self.counter = 0
            #    self.r_sets = {}
        
            # print self.C

    def collect(self, state, action, reward):
        if self.r_sets.has_key((state, action)):
            self.r_sets[(state, action)].append(reward)
        else:
            self.r_sets[(state, action)] = [reward]
        self.counter += 1

    def process_action(self, action):
        self.action = action
        return action

    def process_step(self, observation, reward, done, info):
        n_bins = 8
        n_bins_angle = 10

        # Number of states is huge so in order to simplify the situation
        # we discretize the space to: 10 ** number_of_features
        cart_position_bins = pandas.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1][1:-1]
        pole_angle_bins = pandas.cut([-2, 2], bins=n_bins_angle, retbins=True)[1][1:-1]
        cart_velocity_bins = pandas.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
        angle_rate_bins = pandas.cut([-3.5, 3.5], bins=n_bins_angle, retbins=True)[1][1:-1]

        cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation
        state = build_state([to_bin(cart_position, cart_position_bins),
                             to_bin(pole_angle, pole_angle_bins),
                             to_bin(cart_velocity, cart_velocity_bins),
                             to_bin(angle_rate_of_change, angle_rate_bins)])
        reward = self.noisy_reward(reward)
        # print (reward, "noisy")
        self.collect(state, self.action, reward)
        reward = self.process_reward(reward)

        if self.smooth:
            reward = self.smooth_reward(state, self.action, reward)

        # print (reward, "surrogate")
        return observation, reward, done, info


class SurrogateRewardProcessor(Processor):
    """
    Learning from surrogate reward
    following paper "Learning from noisy labels"
    """
    def __init__(self, e_=0.2, e=0.2, surrogate=False, epsilon=1e-6):
        assert (e_ + e <= 1.0)
        self.e_ = e_
        self.e = e
        self.surrogate = surrogate
        self.epsilon = 1e-6

    def noisy_reward(self, reward):
        n = np.random.random()
        if np.abs(reward - 1.0) < self.epsilon:
            if (n < self.e):
                return -1 * reward
        else:
            if (n < self.e_):
                return -1 * reward
        return reward

    def process_reward(self, reward):
        r = self.noisy_reward(reward) 
        if not self.surrogate:
            return r
        
        if np.abs(r - 1.0) < self.epsilon:
            r_surrogate = ((1 - self.e_) * r + self.e * r) / (1 - self.e_ - self.e)
        else:
            r_surrogate = ((1 - self.e) * r + self.e_ * r) / (1 - self.e_ - self.e)
        return r_surrogate


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
                              target_model_update=1e-2, policy=policy)
        dqn_normal.compile(Adam(lr=1e-3), metrics=['mae'])
        history_normal = dqn_normal.fit(env, nb_steps=10000, visualize=False, verbose=2)
        dqn_normal.save_weights(os.path.join(LOG_DIR, 'dqn_normal_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
        dqn_normal.test(env, nb_episodes=10, visualize=False, verbose=2)
        
        pandas.DataFrame(history_normal.history).to_csv(os.path.join(LOG_DIR, "normal.csv"))

    elif REWARD == "noisy":
        if not SMOOTH:
            processor_noisy = PostProcessor(e_=ERR_N, e=ERR_P, surrogate=False)
        else:
            processor_noisy = PostProcessor(e_=ERR_N, e=ERR_P, smooth=True, surrogate=False)

        # processor_noisy = SurrogateRewardProcessor(e_=ERR_N, e=ERR_P, surrogate=False)
        dqn_noisy = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                             target_model_update=1e-2, policy=policy, processor=processor_noisy)
        dqn_noisy.compile(Adam(lr=1e-3), metrics=['mae'])
        history_noisy = dqn_noisy.fit(env, nb_steps=10000, visualize=False, verbose=2)
        if not SMOOTH:
            dqn_noisy.save_weights(os.path.join(LOG_DIR, 'dqn_noisy_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
            pandas.DataFrame(history_noisy.history).to_csv(os.path.join(LOG_DIR, "noisy.csv"))
        else:
            dqn_noisy.save_weights(os.path.join(LOG_DIR, 'dqn_noisy_smooth_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
            pandas.DataFrame(history_noisy.history).to_csv(os.path.join(LOG_DIR, "noisy_smooth.csv"))

        dqn_noisy.test(env, nb_episodes=10, visualize=False, verbose=2)
        

    elif REWARD == "surrogate":
        print (ERR_P, "!!!!!!!!")
        if not SMOOTH:
            processor_surrogate = PostProcessor(e_=ERR_N, e=ERR_P, smooth=False, surrogate=True)
        else:
            processor_surrogate = PostProcessor(e_=ERR_N, e=ERR_P, smooth=True, surrogate=True)

        # processor_surrogate = SurrogateRewardProcessor(e_=ERR_N, e=ERR_P, surrogate=True)
        dqn_surrogate = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                                 target_model_update=1e-2, policy=policy, processor=processor_surrogate)
        dqn_surrogate.compile(Adam(lr=1e-3), metrics=['mae'])    
        history_surrogate = dqn_surrogate.fit(env, nb_steps=10000, visualize=False, verbose=2)
        if not SMOOTH:
            dqn_surrogate.save_weights(os.path.join(LOG_DIR, 'dqn_surrogate_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
            pandas.DataFrame(history_surrogate.history).to_csv(os.path.join(LOG_DIR, "surrogate.csv"))
        else:
            dqn_surrogate.save_weights(os.path.join(LOG_DIR, 'dqn_surrogate_smooth_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
            pandas.DataFrame(history_surrogate.history).to_csv(os.path.join(LOG_DIR, "surrogate_smooth.csv"))

        dqn_surrogate.test(env, nb_episodes=10, visualize=False, verbose=2)

    else:
        raise NotImplementedError

if __name__ == "__main__":
    train()
