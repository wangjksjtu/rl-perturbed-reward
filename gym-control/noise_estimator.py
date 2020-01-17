import collections
import pandas
import numpy as np
from rl.core import Processor


def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))


def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]


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


class CartpoleProcessor(Processor):
    """
    Learning from perturbed rewards -- CartPole
    step 1 - Estimate the confusion matrices (2 x 2)
    step 2 - Calculate the surrogate rewards
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
        # perturb the true reward
        n = np.random.random()
        
        if np.abs(reward - self.r1) < self.epsilon:
            if (n < self.e_):
                return self.r2
        else:
            if (n < self.e):
                return self.r1

        return reward
    
    def smooth_reward(self, state, action, reward):
        # variance reduction technique (VRT)
        if (state, action) in self.r_smooth:
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
        # calculate the surrogate reward
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
        # estimate the confusion matrix via majority voting
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
                
            self.C = np.array([[1-e_, e_], [e, 1-e]])
            

    def collect(self, state, action, reward):
        if (state, action) in self.r_sets:
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
        self.collect(state, self.action, reward)
        reward = self.process_reward(reward)

        if self.smooth:
            reward = self.smooth_reward(state, self.action, reward)

        return observation, reward, done, info


class CartpoleSurrogateProcessor(Processor):
    """
    Learning from perturbed reward (confusion matrix is known) -- CartPole
    - calculate the surrogate reward directly
    """
    def __init__(self, e_=0.0, e=0.2, surrogate=False, epsilon=1e-6):
        assert (e_ + e < 1.0)
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


class PendulumProcessor(Processor):
    """
    Learning from perturbed rewards -- Pendulum
    step 1 - Estimate the confusion matrices (17 x 17)
    step 2 - Calculate the surrogate rewards
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
        n = np.random.random()
        prob_list.append(n)
        j = sorted(prob_list).index(n)
        reward = -1 * j

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
            self.C = np.zeros((self.M, self.M))
            self.count = [0] * self.M

            for k in self.r_sets.keys():
                freq_count = collections.Counter(self.r_sets[k])

                if self.reverse:
                    truth, count = freq_count.most_common()[-1]
                else:
                    truth, count = freq_count.most_common()[0]

                self.count[int(-truth)] += len(self.r_sets[k])
            print (self.count)

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

            diag = np.diag(self.C)
            anti_diag = np.diag(np.fliplr(self.C))
            log_string("diag: " + np.array2string(diag, formatter={'float_kind':lambda x: "%.5f" % x}))
            log_string("anti_diag:" + np.array2string(anti_diag, formatter={'float_kind':lambda x: "%.5f" % x}))
            log_string("sum: " + np.array2string(np.sum(self.C, axis=1), formatter={'float_kind':lambda x: "%.2f" % x}))

            if is_invertible(self.C):
                self.phi = np.linalg.inv(self.C).dot(self.mmat)
                self.valid = True
            else: self.valid = False

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
            # log_string(str(self.r_sum / float(self.r_counter)))
            self.r_counter = 0
            self.r_sum = 0

        reward = int(np.ceil(reward))
        reward = self.noisy_reward(reward)
        self.collect(state, self.action, reward)
        reward = self.process_reward(reward)

        return observation, reward, done, info


class PendulumSurrogateProcessor(Processor):
    """
    Learning from perturbed reward (confusion matrix is known) -- Pendulum
    - calculate the surrogate reward directly
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
        print (self.cmat.T.shape, self.mmat.shape)
        self.phi = np.linalg.inv(self.cmat).dot(self.mmat)
        print (self.phi.shape)
        self.r_sum = 0
        self.r_counter = 0

    def noisy_reward(self, reward):
        prob_list = list(self.cummat[abs(reward), :])
        n = np.random.random()
        prob_list.append(n)
        j = sorted(prob_list).index(n)
        reward = -1 * j

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


class NAFPendulumProcessor(Processor):
    def process_reward(self, reward):
        # The magnitude of the reward can be important. Since each step yields a relatively
        # high reward, we reduce the magnitude by two orders.
        return reward / 100.