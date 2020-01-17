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
gym.undo_logger_setup()

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
os.system('cp naf_pendulum.py %s' % (LOG_DIR)) # bkp of train procedure
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
        self.phi = np.linalg.inv(self.cmat).dot(self.mmat)
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
            return self.phi[int(-r), 0] / 100.0
        return r / 100.0


class PendulumProcessor(Processor):
    def process_reward(self, reward):
        # The magnitude of the reward can be important. Since each step yields a relatively
        # high reward, we reduce the magnitude by two orders.
        return reward / 100.


def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]

class PostProcessor(Processor):
    """
    Learning from surrogate reward
    following paper "Learning from noisy labels"
    """
    def __init__(self, weight=0.2, surrogate=False, noise_type="norm_one", epsilon=1e-6):
        self.r_sets = {}
        self.weight = weight
        self.surrogate = surrogate

        self.M = 17
        self.cmat, _ = initialize_cmat(noise_type, self.M, self.weight)
        # assert (is_invertible(self.cmat))
        self.cummat = np.cumsum(self.cmat, axis=1)
        self.mmat = np.expand_dims(np.asarray(range(0, -1 * self.M, -1)), axis=1)

        # print self.cmat.T.shape, self.mmat.shape
        # self.phi = np.linalg.inv(self.cmat).dot(self.mmat)
        # print self.phi.shape
        
        self.r_sum = 0
        self.r_counter = 0

        self.counter = 0
        self.C = np.identity(self.M)
        self.epsilon = epsilon

        if self.weight > 0.5:
            self.reverse = True
        else: self.reverse = False
        self.valid = False

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
        if not self.surrogate:
            return reward
        
        self.estimate_C()

        if self.valid:
            return self.phi[int(-reward), 0]
        else: return reward

    def estimate_C(self):
        if self.counter >= 1000 and self.counter % 100 == 0:
            # self.C = np.identity(self.M)
            self.C = np.zeros((self.M, self.M))
            self.count = [0] * self.M
            # self.first_time = [True] * self.M

            for k in self.r_sets.keys():
                freq_count = collections.Counter(self.r_sets[k])
                
                if self.reverse:
                    truth, count = freq_count.most_common()[-1]
                else: 
                    truth, count = freq_count.most_common()[0]
                
                self.count[int(-truth)] += len(self.r_sets[k])
            print self.count

            # print (self.count1, self.count2)
            for k in self.r_sets.keys():
                freq_count = collections.Counter(self.r_sets[k])                    
                list_freq = freq_count.most_common()
                if self.reverse:
                    list_freq = sorted(list_freq, reverse=True)
                truth, count = list_freq[0]
                # if self.first_time[int(-truth)]:
                #     self.C[int(-truth), int(-truth)] = 0
                    # self.first_time[int(-truth)] = False
                # print (prob_correct)

                for pred, count in list_freq:
                    self.C[int(-truth), int(-pred)] += float(count) / self.count[int(-truth)]
                
            # print prob
            diag = np.diag(self.C)
            anti_diag = np.diag(np.fliplr(self.C))
            log_string("diag: " + np.array2string(diag, formatter={'float_kind':lambda x: "%.5f" % x}))
            log_string("anti_diag:" + np.array2string(anti_diag, formatter={'float_kind':lambda x: "%.5f" % x}))
            log_string("sum: " + np.array2string(np.sum(self.C, axis=1), formatter={'float_kind':lambda x: "%.2f" % x}))

            if is_invertible(self.C):
                self.phi = np.linalg.inv(self.C).dot(self.mmat)
                self.valid = True
            else: self.valid = False

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
        # print ("action before:", action)
        n_bins = 20
        action_bins = pandas.cut([-1.0, 1.0], bins=n_bins, retbins=True)[1][1:-1]
        self.action = build_state([to_bin(action, action_bins)])
        # print ("action after:", self.action)

        return action

    def process_step(self, observation, reward, done, info):        
        n_bins = 20
        n_bins_dot = 20

        # Number of states is huge so in order to simplify the situation
        # we discretize the space to: 10 ** number_of_features
        cos_theta_bins = pandas.cut([-1.0, 1.0], bins=n_bins, retbins=True)[1][1:-1]
        sin_theta_bins = pandas.cut([-1.0, 1.0], bins=n_bins, retbins=True)[1][1:-1]
        theta_dot_bins = pandas.cut([-8.0, 8.0], bins=n_bins_dot, retbins=True)[1][1:-1]

        cos_theta, sin_theta, theta_dot = observation
        state = build_state([to_bin(cos_theta, cos_theta_bins),
                             to_bin(sin_theta, sin_theta_bins),
                             to_bin(theta_dot, theta_dot_bins)])

        self.r_sum += reward
        self.r_counter += 1

        if self.r_counter == 200:
            log_string(str(self.r_sum / float(self.r_counter)))
            self.r_counter = 0
            self.r_sum = 0
        
        reward = int(np.ceil(reward))
                        
        # print ("state:", state)
        reward = self.noisy_reward(reward)
        # print (reward, "noisy")
        self.collect(state, self.action, reward)

        reward = self.process_reward(reward)
        # print (reward, "surrogate")
        

        return observation, reward, done, info


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
        processor = PendulumProcessor()
        naf_normal = NAFAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                         memory=memory, nb_steps_warmup=100, random_process=random_process,
                         gamma=.99, target_model_update=1e-3, processor=processor)
        naf_normal.compile(Adam(lr=.00025, clipnorm=1.), metrics=['mae'])
        history_normal = naf_normal.fit(env, nb_steps=150000, visualize=False, verbose=1, nb_max_episode_steps=200)
        naf_normal.save_weights(os.path.join(LOG_DIR, 'naf_normal_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
        naf_normal.test(env, nb_episodes=10, visualize=False, nb_max_episode_steps=200)

        pandas.DataFrame(history_normal.history).to_csv(os.path.join(LOG_DIR, "normal.csv"))

    elif REWARD == "noisy":
        # processor_noisy = SurrogateRewardProcessor(weight=WEIGHT, surrogate=False, noise_type=NOISE_TYPE)
        processor_noisy = PostProcessor(weight=WEIGHT, surrogate=False, noise_type=NOISE_TYPE)
        naf_noisy = NAFAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                             memory=memory, nb_steps_warmup=100, random_process=random_process,
                             gamma=.99, target_model_update=1e-3, processor=processor_noisy)
        naf_noisy.compile(Adam(lr=.00025, clipnorm=1.), metrics=['mae'])
        history_noisy = naf_noisy.fit(env, nb_steps=150000, visualize=False, verbose=1, nb_max_episode_steps=200)
        naf_noisy.save_weights(os.path.join(LOG_DIR, 'naf_noisy_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
        naf_noisy.test(env, nb_episodes=10, visualize=False, nb_max_episode_steps=200)

        pandas.DataFrame(history_noisy.history).to_csv(os.path.join(LOG_DIR, "noisy.csv"))

    elif REWARD == "surrogate":
        # processor_surrogate = SurrogateRewardProcessor(weight=WEIGHT, surrogate=True, noise_type=NOISE_TYPE)
        processor_surrogate = PostProcessor(weight=WEIGHT, surrogate=True, noise_type=NOISE_TYPE)
        naf_surrogate = NAFAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                                 memory=memory, nb_steps_warmup=100, random_process=random_process,
                                 gamma=.99, target_model_update=1e-3, processor=processor_surrogate)
        naf_surrogate.compile(Adam(lr=.00025, clipnorm=1.), metrics=['mae'])
        history_surrogate = naf_surrogate.fit(env, nb_steps=150000, visualize=False, verbose=1, nb_max_episode_steps=200)
        naf_surrogate.save_weights(os.path.join(LOG_DIR, 'naf_surrogate_{}_weights.h5f'.format(ENV_NAME)), overwrite=True)
        naf_surrogate.test(env, nb_episodes=10, visualize=False, nb_max_episode_steps=200)

        pandas.DataFrame(history_surrogate.history).to_csv(os.path.join(LOG_DIR, "surrogate.csv"))

    else:
        raise NotImplementedError


if __name__ == "__main__":
    train()
