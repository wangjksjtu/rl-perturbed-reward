import numpy as np
import collections

def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

def disarrange(a, axis=-1):
    """
    Shuffle `a` in-place along the given axis.

    Apply numpy.random.shuffle to the given axis of `a`.
    Each one-dimensional slice is shuffled independently.
    """
    b = a.swapaxes(axis, -1)
    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
    # so `a` is shuffled in place, too.
    shp = b.shape[:-1]
    for ndx in np.ndindex(shp):
        np.random.shuffle(b[ndx])
    return

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
            disarrange(i_mat)
            print (i_mat)
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

class PongProcessor:
    def __init__(self, weight=0.2, normal=False, surrogate=True, noise_type="norm_one"):
        M = 3
        self.weight = weight
        self.normal = normal
        self.surrogate = surrogate
        self.cmat, _ = initialize_cmat(noise_type, M, self.weight)
        # assert (is_invertible(self.cmat))

        self.cummat = np.cumsum(self.cmat, axis=1)
        print (self.cmat, self.cummat)
        self.mmat = np.expand_dims(np.asarray([-1.0, 0.0, 1.0]), axis=1)
        print (self.mmat)
        self.phi = np.linalg.inv(self.cmat).dot(self.mmat)
        print (self.phi)
        # self.r_sum = 0
        # self.r_counter = 0

    def noisy_reward(self, reward):
        prob_list = list(self.cummat[int(reward+1), :])
        # print prob_list
        n = np.random.random()
        prob_list.append(n)
        # print sorted(prob_list)
        j = sorted(prob_list).index(n)
        # print (n, j)
        reward = j - 1.0
        # print reward
        return reward

    def process_reward(self, reward):
        # self.r_sum += reward

        reward = int(np.ceil(reward))

        if self.normal:
            return reward

        r = self.noisy_reward(reward)
        if self.surrogate:
            return self.phi[int(r + 1.0), 0]
        return r
        # return np.clip(reward, -1., 1.)

    def process_step(self, rewards):
        rewards_new = []
        for reward in rewards:
            reward = self.process_reward(reward)
            rewards_new.append(reward)

        return rewards_new

class BreakoutProcessor:
    def __init__(self, weight=0.2, normal=False, surrogate=True, noise_type="anti_iden"):
        M = 2
        self.weight = weight
        self.normal = normal
        self.surrogate = surrogate
        self.cmat, _ = initialize_cmat(noise_type, M, self.weight)
        # assert (is_invertible(self.cmat))

        self.cummat = np.cumsum(self.cmat, axis=1)
        print (self.cmat, self.cummat)
        self.mmat = np.expand_dims(np.asarray([0.0, 1.0]), axis=1)
        print (self.mmat)
        self.phi = np.linalg.inv(self.cmat).dot(self.mmat)
        print (np.linalg.inv(self.cmat).dot(self.mmat))

        # self.r_sum = 0
        # self.r_counter = 0

    def noisy_reward(self, reward):
        prob_list = list(self.cummat[int(reward), :])
        # print prob_list
        n = np.random.random()
        prob_list.append(n)
        # print sorted(prob_list)
        j = sorted(prob_list).index(n)
        # print (n, j)
        reward = j
        # print reward
        return reward

    def process_reward(self, reward):
        # self.r_sum += reward

        reward = int(np.ceil(reward))

        if self.normal:
            return reward

        r = self.noisy_reward(reward)
        if self.surrogate:
            return self.phi[int(r), 0]
        return r
        # return np.clip(reward, -1., 1.)

    def process_step(self, rewards):
        rewards_new = []
        for reward in rewards:
            reward = self.process_reward(reward)
            rewards_new.append(reward)

        return rewards_new

class BreakoutProcessor2:
    """
    Learning from surrogate reward
    following paper "Learning from noisy labels"
    """
    def __init__(self, weight=0.2, normal=True, surrogate=False, epsilon=1e-6):
        assert (np.abs(weight - 0.5) > epsilon)
        self.normal = normal
        self.e_ = weight
        self.e = weight
        self.surrogate = surrogate
        self.epsilon = 1e-6
        self.r1 = 0
        self.r2 = 1

    def noisy_reward(self, reward):
        n = np.random.random()
        if np.abs(reward - self.r1) < self.epsilon:
            if (n < self.e):
                return self.r2
        else:
            if (n < self.e_):
                return self.r1
        return reward

    def process_reward(self, reward):
        r = self.noisy_reward(reward)
        if not self.surrogate:
            return r

        if np.abs(r - self.r1) < self.epsilon:
            r_surrogate = ((1 - self.e) * self.r1 - self.e_ * self.r2) / (1 - self.e_ - self.e)
        else:
            r_surrogate = ((1 - self.e_) * self.r2 - self.e * self.r1) / (1 - self.e_ - self.e)

        return r_surrogate

    def process_step(self, rewards):
        if self.normal:
            return rewards

        rewards_new = []
        for reward in rewards:
            reward = self.process_reward(reward)
            rewards_new.append(reward)

        return rewards_new

class AtariProcessor:
    def __init__(self, weight=0.1, normal=True, surrogate=False, epsilon=1e-6):
        assert (np.abs(weight - 0.5) > epsilon)
        self.normal = normal
        self.surrogate = surrogate
        self.r_sets = {}
        self.e_ = weight
        self.e = weight

        self.r1 = 0
        self.r2 = 1
        self.counter = 0
        self.C = np.identity(2)
        self.epsilon = epsilon
        if self.e > 0.5:
            self.reverse = True
        else: self.reverse = False

    def noisy_reward(self, reward):
        n = np.random.random()
        if np.abs(reward - self.r1) < self.epsilon:
            if (n < self.e_):
                return self.r2
        else:
            if (n < self.e):
                return self.r1

        return reward
    
    def noisy_rewards(self, rewards):
        noisy_rewards = []
        for r in rewards:
            noisy_rewards.append(self.noisy_reward(r))
        
        return noisy_rewards

    def process_reward(self, reward):
        if not self.surrogate:
            return reward

        self.est_e_ = self.C[0, 1] 
        self.est_e = self.C[1, 0]

        if np.abs(reward - self.r1) < self.epsilon:
            r_surrogate = ((1 - self.est_e) * self.r1 - self.est_e_ * self.r2) / (1 - self.est_e_ - self.est_e)
        else:
            r_surrogate = ((1 - self.est_e_) * self.r2 - self.est_e * self.r1) / (1 - self.est_e_ - self.est_e)

        return r_surrogate
        
    def process_rewards(self, rewards):
        self.estimate_C()

        rewards_new = []
        for r in rewards:
            rewards_new.append(self.process_reward(r))
        
        return rewards_new


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
                    prob_k = float(len(self.r_sets[k])) / self.count1
                    e_ += prob_k * (1 - prob_correct)
                else:
                    # The estimation of e is not accurate! 
                    # In most cases, the predict true-reward is not r0 (in most cases) 
                    # so the numbers of effective samples are small 
                    prob_k = float(len(self.r_sets[k])) / self.count2
                    e += prob_k * (1 - prob_correct)

            w = e_ if self.count1 >= self.count2 else e
            # print (w, abs(w - self.e_))

            self.C = np.array([[1-w, w], [w, 1-w]])

            # if self.counter >= 10000:
            #     self.counter = 0
            #    self.r_sets = {}
        
            # print self.C

    def collect(self, rewards):
        self.r_sets[self.counter % 1000] = rewards
        self.counter += 1

    def process_step(self, rewards):
        # print (rewards.shape)
        if self.normal:
            return rewards
        
        rewards = self.noisy_rewards(rewards)
        self.collect(rewards)
        rewards = self.process_rewards(rewards)
        # print (rewards)

        return rewards
