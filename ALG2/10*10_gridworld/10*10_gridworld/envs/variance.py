import statistics as stats

import numpy as np


class Variance(object):
    def __init__(self, discount, alpha):
        self.returns_sample = dict()
        self.discount_factor = discount
        self.episode_data = []
        self.Variance = np.zeros((100, 4), dtype=float)
        self.update_Variance = np.zeros((100, 4), dtype=float)
        self.visits = np.zeros((100, 4), dtype=float)
        self.mean = np.zeros((100, 4), dtype=float)
        self.states = []
        self.actions = []
        self.rewards = []
        self.alpha = alpha
        self.flag = 0

    def episode_append(self, episode_data):
        # print(episode_data)
        self.states = []
        self.rewards = []
        self.episode_data = episode_data
        for element in self.episode_data:
            self.states.append((element[0], element[1]))
            self.rewards.append(element[2])

        # replacing revisited with "None"  #
        a = set()
        for i, state in enumerate(self.states):
            if state in a:
                self.states[i] = None
            else:
                a.add(state)

        # calculating returns and adding to dictionary
        for j, state in enumerate(self.states):
            if state == None:
                continue
            returns = 0
            rew: float
            for k, rew in enumerate(self.rewards[j:], start=j):
                returns += (self.discount_factor ** (k - j)) * rew
            if state in self.returns_sample:
                self.returns_sample[state].append(returns)
            else:
                self.returns_sample[state] = [returns]
        # print(self.returns_sample)

    def variance_calc(self):
        if self.flag == 0:
            self.flag = 1

            for state in self.returns_sample:
                self.Variance[state] = stats.variance(self.returns_sample[state])
                self.mean[state] = stats.mean(self.returns_sample[state])
                self.visits[state] = len(self.returns_sample[state])
                self.returns_sample[state].clear()
        if self.flag == 1:
            for state in self.returns_sample:
                for i in self.returns_sample[state]:
                    if self.visits[state] == 0:
                        print(state)
                    self.Variance[state] = (((self.visits[state] - 2) * self.Variance[state]) + (i - ((self.mean[state] * self.visits[state] + i) / self.visits[state] + 1)) * (i - self.mean[state])) / (self.visits[state] - 1)
                    self.mean[state] = ((self.mean[state] * self.visits[state]) + i) / (self.visits[state] + 1)
                    self.visits[state] += 1
                self.returns_sample[state].clear()
