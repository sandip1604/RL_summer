import numpy as np
import random
import Gridenv
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, Rectangle
import seaborn as sns
import statistics as stats
import os


def obs_trp_lists():
    obs_lst = []
    for row in range(10):
        for col in range(10):
            if env.GridWorld["state"][row, col] == "obstacle":
                obs_lst.append((col, row))

    # list of traps#
    trp_lst = []
    for row in range(10):
        for col in range(10):
            if env.GridWorld["state"][row][col] == "trap":
                trp_lst.append((col, row))
    return (obs_lst, trp_lst)


env = Gridenv.Grid()


##---------------------------Tabular Q-learning  ----------------------------##

def Q_learning(epsilon1, epsilon2):
    learning_rate = 0.2
    discount_rate = 0.9
    epsilon1 = epsilon1
    epsilon2 = epsilon2

    episode_goal = []
    q_table = np.zeros([100, 4])
    no_episodes = 10000000
    times_goal = 0
    times_traped = 0
    for episodes in range(0, no_episodes):
        terminal_state = 0
        env.reset()

        # env.position = (0,0)
        position = env.position
        state = ((list(position)[0] * 10) + list(position)[1])

        while True:
            ####------------------choosing_action-----------------------########

            if random.uniform(0, 1) < epsilon1:
                chosen_action = random.randint(0, 3)

            else:
                chosen_action = np.argmax(q_table[state, :], axis=0)

            # taking action
            if state == 99:
                print(chosen_action)
            list_next_state = env.step(chosen_action, epsilon2)
            if list_next_state[3]:
                # print("goal reached")
                episode_goal.append(1)
                times_goal = times_goal + 1

            if list_next_state[-1] and (not list_next_state[3]):
                # print("trapped")
                episode_goal.append(0)
                times_traped = times_traped + 1

            reward = list_next_state[2]
            position = copy.deepcopy(list_next_state[0])
            terminal_state = list_next_state[-1]
            next_state = ((list(position)[0] * 10) + list(position)[1])

            # Recalculate
            q_value = q_table[state, chosen_action]

            max_value = np.max(q_table[next_state, :])
            new_q_value = (1 - learning_rate) * q_value + learning_rate * (reward + discount_rate * max_value)
            change = abs(q_value - new_q_value)
            if episodes > 9999990:
                if change >0.2:
                    print(state)
                    print(change)
            q_table[state, chosen_action] = new_q_value
            state = next_state

            if terminal_state:
                break
            # print(q_table)

    return q_table


def get_maxQ_array(q_table):
    list_maxQ = []
    list_maxaction = []
    for i in range(0, 100):
        max_q = np.amax(q_table[i, :], axis=0)
        index = np.argmax(q_table[i,:], axis=0)
        list_maxQ.append(max_q)
        list_maxaction.append(index)
    print(list_maxQ)
    maxQ_array = np.array(list_maxQ)
    maxQ_array = maxQ_array.reshape((10, 10))
    maxQ_action = np.array(list_maxaction)
    maxQ_action = maxQ_action.reshape((10, 10))
    return (maxQ_array, maxQ_action)


def get_variance(q_table):
    list_var = []
    for i in range(0, 100):
        list_var.append(stats.stdev(q_table[i, :]))
    var_array = np.array(list_var)
    var_array = var_array.reshape((10, 10))
    return var_array


epsilon1 = [0.1]
epsilon2 = [0.1]
counter = 0
fig1, ax1 = plt.subplots(1, 2, figsize = (20,10))
#fig2, ax2 = plt.subplots(1, 1)

for i, p in enumerate(epsilon1, 0):
    for j, q in enumerate(epsilon2, 0):
        rows = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        col = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        q_table = Q_learning(p, q)
        (maxQ_array, max_action) = get_maxQ_array(q_table)
        # var_array = get_variance(q_table)
        # var_array = np.around(var_array, decimals=2)
        # var_array = var_array.astype(np.float)
        heatmap_maxQ = sns.heatmap(maxQ_array, annot=True, cmap="YlGnBu", ax=ax1[i])
        heatmap_maxA = sns.heatmap(max_action, annot=True, cmap="YlGnBu", ax=ax1[i+1])
        heatmap_maxQ.set_title("epsilon = %f and p = %f" % (round(p, 2), round(q, 2)))
        heatmap_maxA.set_title("max Q value action heat map: 0 - up, 1 - down, 2 - left, 3 - right")
        # heatmap_var = sns.heatmap(var_array, annot=True, ax=ax2)
        # ax2.set_title("epsilon = %f and p = %f" % (round(p, 2), round(q, 2)))
        # # counter = counter + 1
        obs_lst, trp_lst = obs_trp_lists()

        for r in obs_lst:
            heatmap_maxQ.add_patch(Rectangle(r, 1, 1, fill=False, edgecolor='green', lw=3))
        for s in trp_lst:
            heatmap_maxQ.add_patch(Rectangle(s, 1, 1, fill=True, edgecolor='black', lw=3))
            heatmap_maxQ.add_patch(Rectangle((9, 9), 1, 1, fill=False, edgecolor='red', lw=3))
        for r in obs_lst:
            heatmap_maxA.add_patch(Rectangle(r, 1, 1, fill=False, edgecolor='green', lw=3))
        for s in trp_lst:
            heatmap_maxA.add_patch(Rectangle(s, 1, 1, fill=True, edgecolor='black', lw=3))
            heatmap_maxA.add_patch(Rectangle((9, 9), 1, 1, fill=False, edgecolor='red', lw=3))
fig1.savefig('MaxQ.jpeg', bbox="tight", dpi=200)
fig1.show()
#fig2.savefig('variance.jpeg', bbox="tight", dpi=200)
#fig2.show()

os.system('spd-say "your program has finished"')

# fig2.savefig("hi")



# #### testing the learned policy  ###############

# print("now lets test the policy we have created")
#
# list_steps = []
# goal_steps = []
# trap_steps = []
# test_goal = 0
#
# for episodes in range(0, 100):
#
#     # print("episode number %i running" % episodes)counter
#     terminal_state = 0
#     steps = 0
#     times_traped = 0
#     env.reset()
#     # env.position = (0,0)
#     position = env.position
#     print((position[0] + 1, position[1] + 1))
#     state = ((list(position)[0] * 10) + list(position)[1])
#
#     while True:
#         chosen_action = np.argmax(q_table[state, :], axis=0)
#
#         list_next_state = env.step(chosen_action)
#         steps = steps + 1
#         if list_next_state[3]:
#             print("goal reached")
#             # print(steps)
#             test_goal = test_goal + 1
#         if (list_next_state[-1] and (not list_next_state[3])):
#             # print("trapped")
#             times_traped = times_traped + 1
#
#         position = copy.copy(list_next_state[0])
#         # print(position)
#         terminal_state = list_next_state[-1]
#         next_state = ((list(position)[0] * 10) + list(position)[1])
#         state = next_state
#
#         if terminal_state:
#             break
#     print("while loop exited--------------------------")
# print(test_goal)
# # print(q_table)
# # plt.plot(list_steps, label = 'normal steps')
# #
# # # plt.show()
