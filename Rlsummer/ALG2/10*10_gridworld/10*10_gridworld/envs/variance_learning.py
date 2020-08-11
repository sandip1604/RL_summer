import numpy as np
import math


class Variance(object):
    def __init__(self, discount):
        self.second_moment = np.zeros((100, 4))
        self.discount_rate = discount
        self.Variance = np.zeros((100, 4))
        self.std = np.zeros((100,4))

    def update_secondmoment(self, state_action_reward_pair, J_old, visits):
        (old_state, action_taken, new_state, action_likely, reward) = state_action_reward_pair
        self.second_moment[old_state, action_taken] = self.second_moment[old_state, action_taken] + (1 / visits) * (
                    (reward ** 2) + (2 * self.discount_rate * reward * J_old) + (
                        (self.discount_rate ** 2) * self.second_moment[new_state, action_likely]) - self.second_moment[
                        old_state, action_taken])

    def variance_update(self, state, action, visits, J_new):
        self.Variance[state, action] = self.Variance[state, action] + (1 / visits) * (self.second_moment[state, action] - J_new ** 2 - self.Variance[state, action])

