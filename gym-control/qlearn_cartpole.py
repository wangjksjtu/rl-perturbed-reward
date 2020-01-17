import argparse
import collections
import os
import random

import numpy as np

import gym
import pandas
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--error_positive', type=float, default=0.2,
                    help='Error positive rate [default: 0.2]')
parser.add_argument('--error_negative', type=float, default=0.,
                    help='Error negative rate [default: 0.]')
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

assert(REWARD in ["normal", "noisy", "surrogate"])
if REWARD == "normal":
    LOG_DIR = os.path.join(FLAGS.log_dir, "qlearn_cartpole")
else:
    LOG_DIR = os.path.join(os.path.join(FLAGS.log_dir, "qlearn_cartpole"), str(ERR_P))
ENV_NAME = 'CartPole-v0'

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
os.system('cp qlearn_cartpole.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'setting.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        '''
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))]
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]
        if return_q: # if they want it, give it!
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)

def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]


class SurrogateRewardProcessor():
    """
    Learning from surrogate reward
    following paper "Learning from noisy labels"
    """
    def __init__(self, e_=0.0, e=0.2, surrogate=False, epsilon=1e-6):
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


class PreProcessor:
    "Add noise to reward"
    def __init__(self, e_=0.1, e=0.3, normal=True, epsilon=1e-6):
        assert (np.abs(e_ + e - 1) > epsilon)
        self.normal = normal
        self.e_ = e_
        self.e = e
        self.epsilon = 1e-6
        self.r1 = -1
        self.r2 = 1

    def noisy_reward(self, reward):
        n = np.random.random()
        if np.abs(reward - self.r1) < self.epsilon:
            if (n < self.e_):
                return self.r2
        else:
            if (n < self.e):
                return self.r1
        return reward

    def process_reward(self, reward):
        if self.normal:
            return reward

        r = self.noisy_reward(reward)
        return r

class PostProcessor:
    """
    Learning from surrogate reward
    following paper "Learning from noisy labels"
    """
    def __init__(self, smooth=False, surrogate=True,reverse=False, epsilon=1e-6):
        self.surrogate = surrogate
        self.smooth = smooth
        self.r_sets = {}
        self.r_smooth = {}
        self.r1 = -1
        self.r2 = 1
        self.counter = 0
        self.C = np.identity(2)
        self.epsilon = epsilon
        self.reverse = reverse

    def process_reward(self, reward):
        self.estimate_C()
        self.e_ = self.C[0, 1]
        self.e = self.C[1, 0]

        if self.surrogate:
            if np.abs(reward - self.r1) < self.epsilon:
                reward = ((1 - self.e) * self.r1 - self.e_ * self.r2) / (1 - self.e_ - self.e)
            else:
                reward = ((1 - self.e_) * self.r2 - self.e * self.r1) / (1 - self.e_ - self.e)

        return reward

    def estimate_C(self):
        if self.counter >= 100 and self.counter % 100 == 0:
            e_ = 0; e = 0
            # a = 0; b = 0
            # prob = 0

            self.count1 = 0
            self.count2 = 0
            for k in self.r_sets.keys():
                freq_count = collections.Counter(self.r_sets[k])
                if self.reverse:
                    truth, count = freq_count.most_common()[-1]
                else: truth, count = freq_count.most_common()[0]
                if truth == self.r1:
                    self.count1 += len(self.r_sets[k])
                else:
                    self.count2 += len(self.r_sets[k])

            for k in self.r_sets.keys():
                freq_count = collections.Counter(self.r_sets[k])
                # if self.e_ > 0.05:
                #    self.reverse = True
                #    self.counter = 0; self.r_sets = {}
                #    break

                if self.reverse:
                    truth, count = freq_count.most_common()[-1]
                else:
                    truth, count = freq_count.most_common()[0]
                prob_correct = float(count) / len(self.r_sets[k])
                if truth == self.r1:
                    if self.count1 > 2000:
                        prob_k = float(len(self.r_sets[k])) / self.count1
                        e_ += prob_k * (1 - prob_correct)
                    else: e_ = 0.0
                    # a += 2 * prob_k * prob_correct
                else:
                    prob_k = float(len(self.r_sets[k])) / self.count2
                    e += prob_k * (1 - prob_correct)
                    # b += 2 * prob_k * prob_correct

            # print prob
            log_string(str(e_) + " " + str(e))
            self.C = np.array([[1-e_, e_], [e, 1-e]])

            # if self.counter >= 10000:
            #     self.counter = 0
            #    self.r_sets = {}

            # print self.C

    def smooth_reward(self, state, action, reward):
        if self.smooth:
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

    def collect(self, state, action, reward):
        if self.r_sets.has_key((state, action)):
            self.r_sets[(state, action)].append(reward)
        else:
            self.r_sets[(state, action)] = [reward]
        self.counter += 1


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    goal_average_steps = 195
    max_number_of_steps = 200
    last_time_steps = np.ndarray(0)
    n_bins = 8
    n_bins_angle = 10

    number_of_features = env.observation_space.shape[0]
    last_time_steps = np.ndarray(0)

    # Number of states is huge so in order to simplify the situation
    # we discretize the space to: 10 ** number_of_features
    cart_position_bins = pandas.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1][1:-1]
    pole_angle_bins = pandas.cut([-2, 2], bins=n_bins_angle, retbins=True)[1][1:-1]
    cart_velocity_bins = pandas.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]
    angle_rate_bins = pandas.cut([-3.5, 3.5], bins=n_bins_angle, retbins=True)[1][1:-1]

    # The Q-learn algorithm
    qlearn = QLearn(actions=range(env.action_space.n),
                    alpha=0.5, gamma=0.90, epsilon=0.1)


    pre_processor = PreProcessor(normal=False, e_=ERR_N, e=ERR_P)
    if ERR_P > 0.5:
        if not SMOOTH: post_processor = PostProcessor(smooth=False, surrogate=True, reverse=True)
        else: post_processor = PostProcessor(smooth=True, surrogate=True, reverse=True)

    else:
        if not SMOOTH: post_processor = PostProcessor(smooth=False, surrogate=True)
        else: post_processor = PostProcessor(smooth=True, surrogate=True)

    steps = 0
    while True:
        observation = env.reset()

        cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation
        state = build_state([to_bin(cart_position, cart_position_bins),
                             to_bin(pole_angle, pole_angle_bins),
                             to_bin(cart_velocity, cart_velocity_bins),
                             to_bin(angle_rate_of_change, angle_rate_bins)])

        for t in range(max_number_of_steps):
            # env.render()

            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)
            if REWARD == "noisy":
                reward = pre_processor.process_reward(reward)
                reward = post_processor.smooth_reward(state, action, reward)

            elif REWARD == "surrogate":
                reward = pre_processor.process_reward(reward)
                post_processor.collect(state, action, reward)
                reward = post_processor.process_reward(reward)
                reward = post_processor.smooth_reward(state, action, reward)
            else:
                pass

            # Digitize the observation to get a state
            cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation
            nextState = build_state([to_bin(cart_position, cart_position_bins),
                                     to_bin(pole_angle, pole_angle_bins),
                                     to_bin(cart_velocity, cart_velocity_bins),
                                     to_bin(angle_rate_of_change, angle_rate_bins)])

            # # If out of bounds
            # if (cart_position > 2.4 or cart_position < -2.4):
            #     reward = -200
            #     qlearn.learn(state, action, reward, nextState)
            #     print("Out of bounds, reseting")
            #     break

            if not(done):
                qlearn.learn(state, action, reward, nextState)
                state = nextState
            else:
                # Q-learn stuff
                reward = -20
                qlearn.learn(state, action, reward, nextState)
                last_time_steps = np.append(last_time_steps, [int(t + 1)])
                # print last_time_steps
                break
            steps += 1

        if steps >= 30000: break

    l = last_time_steps.tolist()
    if REWARD == "normal":
        pandas.DataFrame(l).to_csv(os.path.join(LOG_DIR, "normal.csv"))
    elif REWARD == "noisy":
        if not SMOOTH:
            pandas.DataFrame(l).to_csv(os.path.join(LOG_DIR, "noisy.csv"))
        else:
            pandas.DataFrame(l).to_csv(os.path.join(LOG_DIR, "noisy_smooth.csv"))
    else:
        if not SMOOTH:
            pandas.DataFrame(l).to_csv(os.path.join(LOG_DIR, "surrogate.csv"))
        else:
            pandas.DataFrame(l).to_csv(os.path.join(LOG_DIR, "surrogate_smooth.csv"))

    # l.sort()
    # print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    # print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    # env.monitor.close()
