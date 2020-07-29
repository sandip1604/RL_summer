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

plt.style.use('ggplot')

#-------------------------- collecting lis of obstacle and traps -------------------#

def obs_trp_lists(q_table):
    obs_lst = []
    trp_lst = []
    for row in range(10):
        for col in range(10):
            if env.GridWorld["state"][row, col] == "obstacle":
                state = row*10+col
                q_table[state][:] = None
                obs_lst.append((col, row))
            if env.GridWorld["state"][row][col] == "trap":
                state = row * 10 + col
                q_table[state][:] = None
                trp_lst.append((col, row))

    # list of traps#
    return (obs_lst, trp_lst)

##---------------------------Tabular Q-learning  ----------------------------##



def Q_learning(epsilon1, epsilon2):
    learning_rate = 0.2
    discount_rate = 0.9
    epsilon1 = epsilon1
    epsilon2 = epsilon2
    Qtable_list = []
    episode_goal = []
    q_table = np.random.rand(100, 5) / 10000
    no_episodes = 10000000
    for episodes in range(0, no_episodes):
        if (episodes % 1000000) == 0:
            os.system('spd-say "your program has reached one milestone"')
        terminal_state = 0
        env.reset()
        try:
            log_results = math.log(episodes, 4)
            if log_results.is_integer() or episodes == 9999999:
                Qtable_list.append(copy.deepcopy(q_table))
                print(episodes)
        except:
            print(episodes)

        # if random.uniform(0, 1) < 0.7:
        #     while True:
        #         col = random.randint(5, 9)
        #         row = random.randint(0, 4)
        #         if env.GridWorld['state'][row][col] == S:
        #             env.position = (row, col)
        #             break
        # else:
        #     while True:
        #         col = random.randint(0, 4)
        #         row = random.randint(0, 4)
        #         if env.GridWorld['state'][row][col] == S:
        #             env.position = (row, col)
        #             break

        # env.position = (0,0)
        position = env.position
        state = ((list(position)[0] * 10) + list(position)[1])

        while True:
            ####------------------choosing_action-----------------------########



            if random.uniform(0, 1) < epsilon1:
                chosen_action = random.randint(0, 3)

            else:

                chosen_action = np.argmax(q_table[state, :], axis=0)

            if chosen_action == 4:
                action = env.GridWorld["expert"][position[0]][position[1]]
                expert_call = 1

            else:
                expert_call = 0
                action = chosen_action

            list_next_state = env.step(action, epsilon2, expert_call)

            reward = list_next_state[2]
            position = copy.deepcopy(list_next_state[0])
            terminal_state = list_next_state[-1]
            next_state = ((list(position)[0] * 10) + list(position)[1])

            # Recalculate
            q_value = q_table[state, chosen_action]

            max_value = np.max(q_table[next_state, :])
            new_q_value = (1 - learning_rate) * q_value + learning_rate * (reward + discount_rate * max_value)

            q_table[state, chosen_action] = new_q_value
            state = next_state

            if terminal_state:
                break
            # print(q_table)

    return q_table, Qtable_list

#------------ getting arrays for max Q and varriance ------------------#

def get_maxQ_array(q_table):
    list_maxQ = []
    list_max_action = []
    for i in range(0, 100):
        if np.isnan(q_table[i, 0]):
            list_maxQ.append(np.nan)
            list_max_action.append(np.nan)
        else:
            max_q = np.amax(q_table[i, :], axis=0)
            max_a = np.argmax(q_table[i, :], axis=0)
            list_max_action.append(max_a)
            list_maxQ.append(max_q)



    maxQ_array = np.array(list_maxQ)
    maxQ_array = maxQ_array.reshape((10, 10))
    maxA_array = np.array(list_max_action)
    maxA_array = maxA_array.reshape((10, 10))
    return (maxQ_array, maxA_array)


def get_variance_array(q_table):
    list_var = []
    for i in range(0, 100):
        list_var.append(stats.stdev(q_table[i, :]))
    var_array = np.array(list_var)
    var_array = var_array.reshape((10, 10))
    return var_array

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


if __name__ == "__main__":

    S = "safe"
    env = Gridenv.Grid()
    epsilon1 = [0.1]
    epsilon2 = [0.3]
    counter = 0
    fig1, ax1 = plt.subplots(1, 2, figsize=(20, 10))
    fig2, ax2 = plt.subplots(1, 2, figsize=(20, 10))

    for i, p in enumerate(epsilon1, 0):
        for j, q in enumerate(epsilon2, 0):
            rows = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            col = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            q_table, Qtable_list = Q_learning(p, q)
            obs_lst, trp_lst = obs_trp_lists(q_table)
            (maxQ_array, maxA_array) = get_maxQ_array(q_table)
            var_array = get_variance_array(q_table)
            var_array = np.around(var_array, decimals=2)
            var_array = var_array.astype(np.float)
            maxQ_array = maxQ_array.astype(np.float)
            np.set_printoptions(formatter={'float_kind': '{:f}'.format})
            heatmap_max = sns.heatmap(maxQ_array, annot=True, cmap="YlGnBu", ax=ax1[0], fmt=".1f")
            ax1[0].set_title("epsilon = %f and p = %f" % (round(p, 2), round(q, 2)))
            heatmap_action1 = sns.heatmap(maxA_array, annot=True, cmap="YlGnBu", ax=ax1[1])
            ax1[1].set_title("0- up , 1 - down, 2 - left, 3 - right")
            heatmap_var = sns.heatmap(var_array, annot=True, ax=ax2[0], fmt=".1f")
            ax2[0].set_title("epsilon = %f and p = %f" % (round(p, 2), round(q, 2)))
            heatmap_action2 = sns.heatmap(maxA_array, annot=True, cmap="YlGnBu", ax=ax2[1])
            ax1[1].set_title("0- up , 1 - down, 2 - left, 3 - right")

            for r in obs_lst:
                heatmap_max.add_patch(Rectangle(r, 1, 1, fill=False, edgecolor='green', lw=3))
                heatmap_var.add_patch(Rectangle(r, 1, 1, fill=False, edgecolor='green', lw=3))
                heatmap_action1.add_patch(Rectangle(r, 1, 1, fill=False, edgecolor='green', lw=3))
                heatmap_action2.add_patch(Rectangle(r, 1, 1, fill=False, edgecolor='green', lw=3))
            for s in trp_lst:
                heatmap_max.add_patch(Rectangle(s, 1, 1, fill=True, color = "blue" , edgecolor='black', lw=3))
                heatmap_max.add_patch(Rectangle((9, 9), 1, 1, fill=False, edgecolor='red', lw=3))
                heatmap_var.add_patch(Rectangle(s, 1, 1, fill=True, color = "blue" , edgecolor='black', lw=3))
                heatmap_var.add_patch(Rectangle((9, 9), 1, 1, fill=False, edgecolor='red', lw=3))
                heatmap_action2.add_patch(Rectangle(s, 1, 1, fill=True, color = "blue" , edgecolor='black', lw=3))
                heatmap_action2.add_patch(Rectangle((9, 9), 1, 1, fill=False, edgecolor='red', lw=3))
                heatmap_action1.add_patch(Rectangle(s, 1, 1, fill=True, color = "blue" , edgecolor='black', lw=3))
                heatmap_action1.add_patch(Rectangle((9, 9), 1, 1, fill=False, edgecolor='red', lw=3))

    fig1.savefig('MaxQ-ALG1-10M-0.3.png', bbox="tight", dpi=200)
    fig1.show()
    fig2.savefig('variance-ALG1-10M-0.3.png', bbox="tight", dpi=200)
    fig2.show()

    #-------------------- CHECKING THE POLICY WITH REWARDS GRAPH------#


    start_points = [(1, 1), (2, 7), (4, 3), (6, 3), (0, 0), (0,4), (8,5), (5,8), (6,0), (8,0), (3,7)]

    avg_reward_list = []
    for i in Qtable_list:
        reward_q_list = []
        for p in range(0, 8):
            iter_reward_list = []
            for j in start_points:

                env.reset()
                env.position = j
                position = env.position
                state = ((list(position)[0] * 10) + list(position)[1])
                return_recieved = 0
                while True:
                    ####------------------choosing_action-----------------------########
                    chosen_action = np.argmax(i[state, :], axis=0)

                    if chosen_action == 4:
                        action = env.GridWorld["expert"][position[0]][position[1]]
                        expert_call = 1

                    else:
                        expert_call = 0
                        action = chosen_action

                    list_next_state = env.step(action, epsilon2[0], expert_call)

                    reward = list_next_state[2]
                    position = copy.deepcopy(list_next_state[0])
                    terminal_state = list_next_state[-1]
                    next_state = ((list(position)[0] * 10) + list(position)[1])
                    return_recieved += reward
                    state = next_state

                    if terminal_state:
                        iter_reward_list.append(return_recieved)
                        break
            reward_q_list = reward_q_list+iter_reward_list
        #if stats.mean(reward_q_list) >-1000:
        avg_reward_list.append(stats.mean(reward_q_list))
    avg_reward_list = [round(num, 1) for num in avg_reward_list]
    fig, ax = plt.subplots()
    fig1 = ax.bar(torch.arange(len(avg_reward_list)), avg_reward_list, alpha=0.6, color='green')
    ax.set_title('return collected')
    ax.set_xlabel("number of episodes of training ( in log scale to the base 4)")
    autolabel(fig1)
    plt.savefig("reward_Qlearning-ALG1-3")
    plt.show()

# fig2.savefig("hi")

# ---------------- ploting heatmap --------------------#

# rows = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
# col = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
#
# sns.set(style = "whitegrid")
# heat_map = sns.heatmap(maxQ_array, annot= True, cmap="YlGnBu")
#
# # list of obstacle#
# plt.show()


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
