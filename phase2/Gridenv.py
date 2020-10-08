import random
from typing import Any, Union

import gym
from gym import error, spaces, utils, Env
from gym.utils import seeding
import copy

import numpy as np

# from six import StringIO, b
#random.seed(23)

up = 0
down = 1
left = 2
right = 3

S = "safe"
T = "trap"
O = "obstacle"
G = "goal"


class Grid(gym.Env):
    # metadata = {'render.modes': ['human']}

    S = "safe"
    T = "trap"
    O = "obstacle"
    G = "goal"

    def not_valid_actions(self):
        for i in range(0, 10):
            for j in range(0, 10):
                #print(i, j)
                if not (self.GridWorld["state"][i][j] == O or self.GridWorld["state"][i][j] == T):
                    if (i + 1) == 10:
                        if self.GridWorld["state"][i - 1][j] == O:
                            self.not_valid_action["up"].add((i, j))
                    else:
                        if self.GridWorld["state"][i + 1][j] == O:
                            self.not_valid_action["down"].add((i, j))
                        if self.GridWorld["state"][i - 1][j] == O:
                            self.not_valid_action["up"].add((i, j))

                    if (j + 1) == 10:
                        if self.GridWorld["state"][i][j - 1] == O:
                            self.not_valid_action["left"].add((i, j))
                    else:
                        if self.GridWorld["state"][i][j + 1] == O:
                            self.not_valid_action["right"].add((i, j))
                        if self.GridWorld["state"][i][j - 1] == O:
                            self.not_valid_action["left"].add((i, j))

    def __init__(self):
        up = 0
        down = 1
        left = 2
        right = 3

        S = "safe"
        T = "trap"
        O = "obstacle"
        G = "goal"
        self.GridWorld = {"state": np.array([S, S, S, S, S, S, S, S, S, S,
                                             S, S, S, S, S, S, O, O, O, S,
                                             S, S, T, T, S, S, S, S, O, S,
                                             S, S, T, T, S, S, S, S, S, S,
                                             S, S, S, S, S, S, S, S, S, S,
                                             S, S, S, S, T, T, T, S, S, S,
                                             S, S, O, S, T, T, T, S, S, S,
                                             S, S, O, S, S, S, S, S, S, S,
                                             S, O, O, O, S, S, S, S, S, S,
                                             S, S, S, S, S, S, S, S, S, G,]).reshape(10, 10),
                          "expert": np.array([1, 3, 3, 3, 3, 3, 3, 3, 3, 1,
                                              1, 2, 2, 3, 3, 1, 1, 1, 1, 1,
                                              1, 2, 0, 0, 3, 1, 1, 1, 1, 1,
                                              1, 2, 0, 0, 3, 3, 3, 3, 1, 1,
                                              1, 1, 1, 1, 3, 0, 3, 3, 1, 1,
                                              1, 2, 3, 1, 0, 0, 0, 3, 1, 1,
                                              1, 2, 0, 1, 0, 0, 0, 3, 1, 1,
                                              1, 2, 0, 3, 1, 1, 1, 3, 1, 1,
                                              1, 0, 0, 0, 3, 1, 1, 1, 1, 1,
                                              3, 3, 3, 3, 3, 3, 3, 3, 3, 0,]).reshape(10,10)
                          }
        self.not_valid_action = {
            "up": {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9)},

            "down": {(9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9)},

            "left": {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0)},

            "right": {(0, 9), (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (8, 9), (9, 9)}
        }
        self.not_valid_actions()
        self.reward = 0
        self.goal = (9, 9)
        self.goal_reached = 0
        self.terminal_state = 0

        while True:
            column = random.randint(0, 9)
            row = random.randint(0, 9)

            if self.GridWorld['state'][row][column] == S:
                self.position = (row, column)
                break

    def step(self, action, epsilon2, expert_call):
        return_list = []
        # print(self.position)
        S = "safe"
        T = "trap"
        O = "obstacle"
        G = "goal"
        reg_reward = -2
        goal_reward = 100
        trap_reward = -100
        obstacle_reward = -10
        expert_reward = 0
        total_action: set = {0, 1, 2, 3}
        epsilon2 = epsilon2
        if expert_call:
            expert_reward = -5
        else:
            if random.uniform(0, 1) < epsilon2:
                action = random.choice(list(total_action - {action}))
                #print("taken detour:", action)

        if action == 0:

            #print(self.position, "up")
            (row, col) = self.position
            if self.position in self.not_valid_action["up"]:
                #print("can't move up")
                self.reward = obstacle_reward+expert_reward
                # print(self.position,"up")
                return_list = copy.deepcopy(
                    [self.position, self.GridWorld["state"][row][col], self.reward, self.goal_reached,
                     self.terminal_state])
            else:

                self.position = (row - 1, col)
                row, col = self.position
                #print("after up",self.position)
                # checking for terminal state
                if self.position == self.goal:
                    self.goal_reached = 1
                    self.terminal_state = 1
                    self.reward = goal_reward+expert_reward
                    return_list = copy.deepcopy(
                        [self.position, self.GridWorld["state"][row][col], self.reward, self.goal_reached,
                         self.terminal_state])
                    #self.reset()
                elif self.GridWorld["state"][row][col] == T:
                    self.reward = trap_reward+expert_reward
                    self.terminal_state = 1
                    return_list = copy.deepcopy(
                        [self.position, self.GridWorld["state"][row][col], self.reward, self.goal_reached,
                         self.terminal_state])
                    #self.reset()
                else:
                    self.reward = reg_reward+expert_reward
                    # print(self.position)
                    return_list = copy.deepcopy(
                        [self.position, self.GridWorld["state"][row][col], self.reward, self.goal_reached,
                         self.terminal_state])


        elif action == 1:
            #print(self.position, "down")
            (row, col) = self.position
            if self.position in self.not_valid_action["down"]:
                #print("cant move down")
                self.reward = obstacle_reward+expert_reward
                # print(self.position, "down")
                return_list = copy.deepcopy(
                    [self.position, self.GridWorld["state"][row][col], self.reward, self.goal_reached,
                     self.terminal_state])
            else:

                self.position = (row + 1, col)
                row, col = self.position
                #print("after down", self.position)
                # checking for terminal state
                if self.position == self.goal:
                    self.goal_reached = 1
                    self.terminal_state = 1
                    self.reward = goal_reward+expert_reward

                    return_list = copy.deepcopy(
                        [self.position, self.GridWorld["state"][row][col], self.reward, self.goal_reached,
                         self.terminal_state])
                    #self.reset()
                elif self.GridWorld["state"][row][col] == T:
                    self.reward = trap_reward+expert_reward
                    self.terminal_state = 1
                    return_list = copy.deepcopy(
                        [self.position, self.GridWorld["state"][row][col], self.reward, self.goal_reached,
                         self.terminal_state])
                    #self.reset()
                else:
                    self.reward = reg_reward+expert_reward
                    # print(self.position)
                    return_list = copy.deepcopy(
                        [self.position, self.GridWorld["state"][row][col], self.reward, self.goal_reached,
                         self.terminal_state])

        elif action == 2:
            #print(self.position, "left")
            (row, col) = self.position
            if self.position in self.not_valid_action["left"]:
                #print("can't move left")
                self.reward = obstacle_reward+expert_reward
                return_list = copy.deepcopy(
                    [self.position, self.GridWorld["state"][row][col], self.reward, self.goal_reached,
                     self.terminal_state])
            else:

                self.position = (row, col - 1)
                row, col = self.position
                #print("after left", self.position)
                # checking for terminal state
                if self.position == self.goal:
                    self.goal_reached = 1
                    self.terminal_state = 1
                    self.reward = goal_reward+expert_reward
                    return_list = copy.deepcopy(
                        [self.position, self.GridWorld["state"][row][col], self.reward, self.goal_reached,
                         self.terminal_state])
                    #self.reset()
                elif self.GridWorld["state"][row][col] == T:
                    self.reward = trap_reward+expert_reward
                    self.terminal_state = 1
                    return_list = copy.deepcopy(
                        [self.position, self.GridWorld["state"][row][col], self.reward, self.goal_reached,
                         self.terminal_state])
                    #self.reset()
                else:
                    self.reward = reg_reward+expert_reward
                    # print(self.position)
                    return_list = copy.deepcopy(
                        [self.position, self.GridWorld["state"][row][col], self.reward, self.goal_reached,
                         self.terminal_state])

        elif action == 3:
            #print(self.position, "right")
            (row, col) = self.position
            if self.position in self.not_valid_action["right"]:
                #print("cant move right")
                self.reward = obstacle_reward+expert_reward
                return_list = copy.deepcopy(
                    [self.position, self.GridWorld["state"][row][col], self.reward, self.goal_reached,
                     self.terminal_state])

            else:

                self.position = (row, col + 1)
                row, col = self.position
                #print("after right", self.position)
                # checking for terminal state
                if self.position == self.goal:
                    self.goal_reached = 1
                    self.terminal_state = 1
                    self.reward = goal_reward+expert_reward
                    return_list = copy.deepcopy(
                        [self.position, self.GridWorld["state"][row][col], self.reward, self.goal_reached,
                         self.terminal_state])

                    #self.reset()
                elif self.GridWorld["state"][row][col] == T:
                    self.reward = trap_reward+expert_reward
                    self.terminal_state = 1
                    return_list = copy.deepcopy(
                        [self.position, self.GridWorld["state"][row][col], self.reward, self.goal_reached,
                         self.terminal_state])

                    #self.reset()
                else:
                    self.reward = reg_reward+expert_reward
                    # print(self.position)
                    return_list = copy.deepcopy(
                        [self.position, self.GridWorld["state"][row][col], self.reward, self.goal_reached,
                         self.terminal_state])
        (row, col) = self.position




        return return_list

    def reset(self):

        S = "safe"
        T = "trap"
        O = "obstacle"
        G = "goal"

        self.reward = 0
        self.goal = (9, 9)
        self.goal_reached = 0
        self.terminal_state = 0
        while True:
            column = random.randint(0, 9)
            row = random.randint(0, 9)

            if self.GridWorld['state'][row][column] == S:
                self.position = (row, column)
                #print(self.position)
                break
        #print(self.position)
        return self.position          # remove for tabular Q_learning
    # def render(self):

    # def close(self):
