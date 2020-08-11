import numpy as np
import random
import Gridenv
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, Rectangle
import seaborn as sns
import statistics as stats
import os
import math
import torch
from variance_learning import Variance
import time
from numba import prange, njit


plt.style.use('ggplot')
def returns_graph(Qtable, variance_table, bound):
    """

    :type Qtable: object
    """
    start_points = [(1, 1), (2, 7), (4, 3), (6, 3), (0, 0), (0, 4), (8, 5), (5, 8), (6, 0), (8, 0), (3, 7)]
    avg_reward_list = []
    traps_list = []
    for i in range(1000):
        env.reset()
        steps = 0
        position = env.position
        state = (list(position)[0] * 10) + list(position)[1]
        return_recieved = 0
        #print(state)
        #print(type(Qtable))
        while True:
            ####------------------choosing_action-----------------------########
            if variance_table[state,np.argmax(Qtable[state,:], axis=0)] > bound:
                chosen_action = 4
            else:
                chosen_action = np.argmax(Qtable[state, :], axis=0)

            if chosen_action == 4:
                action = env.GridWorld["expert"][position[0]][position[1]]
                # print(postion[0], position[1])
                expert_call = 1

            else:
                expert_call = 0
                action = chosen_action

            list_next_state = env.step(action, epsilon2[0], expert_call)
            steps += 1
            #ultimate_list.append((list_next_state[0], action))
            reward = list_next_state[2]
            position = copy.deepcopy(list_next_state[0])
            terminal_state = list_next_state[-1]
            next_state = ((list(position)[0] * 10) + list(position)[1])
            return_recieved += reward
            state = next_state

            if terminal_state:
                #print("entered terminal loop condition")
                #iter_reward_list.append(return_recieved)
                avg_reward_list.append(return_recieved)
                #print(i)
                #print(reward)
                if reward == -100:
                    #print("entering beause reward is -100")
                    traps_list.append(i)
                break




    avg_reward_list = [round(num, 1) for num in avg_reward_list]
    avg_reward_list = stats.mean(avg_reward_list)
    # fig1, ax = plt.subplots()
    # fig1 = ax.bar(torch.arange(len(avg_reward_list)), avg_reward_list, alpha=0.6, color='green')
    # ax.set_xlabel("number of episodes of training ( in log scale to the base 4)")
    # autolabel(fig1, ax)
    # #plt.savefig("reward_Qlearning-ALG2")
    # plt.show()
    #print(traps_list)
    print(bound,"-",len(traps_list))
    return  avg_reward_list, len(traps_list)

def optimise(q_table, variance_table):
    bounds, step = list(np.linspace(0, 600, num=600, endpoint = False, retstep = True))
    print(bounds)
    return_list = []
    wasted_episodes_list = []
    for bound in bounds:
        return_got, episodes_wasted = returns_graph(q_table, variance_table, bound)
        return_list.append(return_got)
        wasted_episodes_list.append(episodes_wasted)
    #print(return_list)
    #print(wasted_episodes_list)
    max = np.amax(return_list)
    min = np.amin(wasted_episodes_list)
    min_index = np.argmin(wasted_episodes_list)
    max_index = np.argmax(return_list)
    return_list = [round(num, 2) for num in return_list]
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig1 = ax1.bar(bounds, return_list, alpha=0.6, color='green')
    fig2 = ax2.bar(bounds, wasted_episodes_list, alpha = 0.6, color = 'red')
    # ax.set_title('rewards collected- ALG2-0.3')
    ax1.set_xlabel("thresshold")
    ax2.set_xlabel("thresshold")
    for rect in fig1:
        height = rect.get_height()
        if height == max:
            ax1.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    plt.savefig("optimisation curve")
    plt.show()
    print(bounds[max_index])
    print(max)
    print(bounds[min_index])
    print(min)

if __name__ == "__main__":
    S = "safe"
    env = Gridenv.Grid()
    ##### PARAMETERS #####
    epsilon1 = [0.1]
    epsilon2 = [0.1]
    learning_rate = 0.2
    discount_rate = 0.9
    no_episodes = 10000000
    counter = 0
    alpha = 0.2
    Q_table = torch.load("q_table_tensor.pt")
    var_table = torch.load("variance_tensor.pt")
    Q_table = Q_table.mean(0)
    Variance_table = var_table.mean(0)
    # Q_table = np.array(Q_table.numpy(),dtype=float).reshape(100,4)
    # Variance_table = Variance_table.numpy()
    print(Q_table.shape)
    optimise(Q_table,Variance_table)
    #returns_graph(Q_table)
