import numpy as np
import random
import Gridenv
import copy
import matplotlib.pyplot as plt

env = Gridenv.Grid()
###----------------- random steps taken-----------------------------####

# no_episodes = 1000
# env = Gridenv.Grid()
# list_steps = []
# for i in range(no_episodes):
#     terminal_state = 0
#     steps = 0
#     env.reset()
#
#     while True:
#         action = random.randint(0, 3)
#         list_1 = env.step(action)
#         steps = steps + 1
#         # print(steps)
#         # print(list_1)
#         # print(list_1[0])
#         if list_1[3] == 1:
#             print(i)
#         if list_1[4] == 1:
#             break
#     # print("-----------------------")
#     list_steps.append(steps)
#
# #plt.plot(list_steps)
# #plt.ylabel("number of steps per episodes")
# #plt.show()

##---------------------------Tabular Q-learning  ----------------------------##
learning_rate = 0.2
discount_rate = 0.9
epsilon = 0.6
episode_goal = []

q_table = np.zeros([100, 4])
no_episodes = 10000
times_goal = 0
times_traped = 0

for episodes in range(0, no_episodes):
    terminal_state = 0
    steps = 0
    env.reset()
    env.position = (0,0)
    position = env.position
    state = ((list(position)[0] * 10) + list(position)[1])

    while True:
        ####-------------------choosing_action-----------------------########

        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(q_table[state, :], axis=0)

        # taking action
        list_next_state = env.step(action)
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
        q_value = q_table[state, action]
        max_value = np.max(q_table[next_state, :])
        new_q_value = (1 - learning_rate) * q_value + learning_rate * (reward + discount_rate * max_value)
        q_table[state, action] = new_q_value
        state = next_state

        if terminal_state:
            break
# print(q_table)
print(times_goal)
print(times_traped)

#### testing the learned policy  ###############

print("now lets test the policy we have created")

list_steps = []
goal_steps = []
trap_steps = []
test_goal = 0

for episodes in range(0, 100):

    # print("episode number %i running" % episodes)
    terminal_state = 0
    steps = 0
    env.position = (0,0)
    position = env.position
    state = ((list(position)[0] * 10) + list(position)[1])

    while True:
        action = np.argmax(q_table[state, :], axis=0)
        # taking action
        list_next_state = env.step(action)
        steps = steps + 1
        if list_next_state[3]:
            print("goal reached")
            print(steps)
            test_goal = test_goal + 1
            times_goal = times_goal + 1
        if (list_next_state[-1] and (not list_next_state[3])):
            print("trapped")
            times_traped = times_traped + 1

        position = copy.copy(list_next_state[0])
        terminal_state = list_next_state[-1]
        next_state = ((list(position)[0] * 10) + list(position)[1])

        # # Recalculate
        # q_value = q_table[state, action]
        # max_value = np.max(q_table[next_state, :])
        # new_q_value = (1 - learning_rate) * q_value + learning_rate * (reward + discount_rate * max_value)
        # q_table[state, action] = new_q_value
        state = next_state

        if terminal_state:
            break
    print("while loop exited")
print(test_goal)
print(q_table)
# plt.plot(list_steps, label = 'normal steps')

# plt.show()
