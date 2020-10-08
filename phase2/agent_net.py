import random
import Gridenv
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, Rectangle
import seaborn as sns
# from variance_learning import Variance
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import random
import torch.nn.functional as F
import os.path

plt.style.use('ggplot')

use_cuda = torch.cuda.is_available()

#device = torch.device("cuda:0" if use_cuda else "cpu")
print(torch.device())
Tensor = torch.Tensor
LongTensor = torch.LongTensor

env = Gridenv.Grid()

##### Params #########################

learning_rate = 0.003
gamma = 0.9
batch_size = 100
replay_mem_size = 50000

epsilon1 = 0.1
epsilon2 = 0.1

hidden_layer1 = 20
hidden_layer2 = 70
hidden_layer3 = 20

number_of_output = 4
# epsilon1 = 0.1
# epsilon2 = 0.3
# learning_rate = 0.2
discount_rate = 0.9
no_episodes = 400
counter = 0
alpha = 0.2
expert_call = 0


#################################################

class ExperienceReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)

        if self.position >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))

    def __len__(self):
        return len(self.memory)


class NeuralNetwork_Q(nn.Module):
    def __init__(self):
        super(NeuralNetwork_Q, self).__init__()

        self.linear1 = nn.Linear(2, hidden_layer1)
        self.linear2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.linear3 = nn.Linear(hidden_layer2, hidden_layer3)
        self.linear4 = nn.Linear(hidden_layer3, number_of_output)
        self.dropout = nn.Dropout(p=0.2)
        # self.btn2 = nn.BatchNorm1d(hidden_layer2)
        # self.btn1 = nn.BatchNorm1d(2)
        # self.activation = nn.Tanh()
        self.activation = nn.ReLU()

    def forward(self, x):
        # code to normalise the input
        x = x - x.mean()
        if not self.train():
            output_linear = self.dropout(self.linear1(x))
            output_linear = self.activation(output_linear)
            output_linear = self.dropout(self.linear2(output_linear))
            output_linear = self.activation(output_linear)
            output_linear = self.dropout(self.linear3(output_linear))
            output_linear = self.linear4(self.activation(output_linear))
            # output_linear = F.softmax(output_linear, dim=1)              # included in loss function

            output_final = output_linear

        else:
            output_linear = self.linear1(x)
            output_linear = self.activation(self.dropout(output_linear))
            output_linear = self.linear2(output_linear)
            output_linear = self.activation(self.dropout(output_linear))
            output_linear = self.linear3(output_linear)
            output_linear = self.linear4(self.activation(self.dropout(output_linear)))
            # output_linear = F.softmax(output_linear, dim=1)              # included in loss function

            output_final = output_linear

        return output_final


class NeuralNetwork_V(nn.Module):
    def __init__(self):
        super(NeuralNetwork_V, self).__init__()

        self.linear1 = nn.Linear(2, hidden_layer1)
        self.linear2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.linear3 = nn.Linear(hidden_layer2, hidden_layer3)
        self.linear4 = nn.Linear(hidden_layer3, number_of_output)

        # self.btn2 = nn.BatchNorm1d(hidden_layer2)
        # self.btn1 = nn.BatchNorm1d(2)
        # self.activation = nn.Tanh()
        self.activation = nn.ReLU()

    def forward(self, x):
        # code to normalise the input
        x = x - x.mean()

        output_linear = self.linear1(x)
        output_linear = self.activation(output_linear)
        output_linear = self.linear2(output_linear)
        output_linear = self.activation(output_linear)
        output_linear = self.linear3(output_linear)
        output_linear = self.linear4(self.activation(output_linear))
        # output_linear = F.softmax(output_linear, dim=1)              # included in loss function

        output_final = output_linear

        return output_final


class NeuralNetwork_SM(nn.Module):
    def __init__(self):
        super(NeuralNetwork_SM, self).__init__()

        self.linear1 = nn.Linear(2, hidden_layer1)
        self.linear2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.linear3 = nn.Linear(hidden_layer2, hidden_layer3)
        self.linear4 = nn.Linear(hidden_layer3, number_of_output)

        self.btn2 = nn.BatchNorm1d(hidden_layer2)
        self.btn1 = nn.BatchNorm1d(2)
        # self.activation = nn.Tanh()
        self.activation = nn.ReLU()

    def forward(self, x):
        # code to normalise the input
        x = x - x.mean()

        output_linear = self.linear1(x)
        output_linear = self.activation(output_linear)
        output_linear = self.linear2(output_linear)
        output_linear = self.activation(output_linear)
        output_linear = self.linear3(output_linear)
        output_linear = self.linear4(self.activation(output_linear))
        # output_linear = F.softmax(output_linear, dim=1)              # included in loss function

        output_final = output_linear

        return output_final


class Agent(object):
    def __init__(self):
        self.Qnn = NeuralNetwork_Q().to(device)
        self.SMnn = NeuralNetwork_SM().to(device)
        self.Vnn = NeuralNetwork_V().to(device)


        for p in self.Qnn.parameters():
            p.data.fill_(random.gauss(0.1, 0.05))
            print(p)

        self.loss_funcSMnn = nn.MSELoss()  # for discrete action space
        self.loss_funcVnn = nn.MSELoss()  # for discrete action space
        self.loss_funcQnn = nn.MSELoss()  # for discrete action space

        self.optimizer_Qnn = optim.Adam(params=self.Qnn.parameters(), lr=learning_rate)
        self.optimizer_SMnn = optim.Adam(params=self.SMnn.parameters(), lr=learning_rate)
        self.optimizer_Vnn = optim.Adam(params=self.Vnn.parameters(), lr=learning_rate)

    def action_selection(self, state, epsilon):

        random_for_egreedy = torch.rand(1).item()
        #print(random_for_egreedy)
        set_action = {0, 1, 2, 3}
        # print(state)
        with torch.no_grad():
            state = Tensor(state).to(device)
            action_from_nn = self.Qnn(state)
            #print(action_from_nn)
            action = torch.max(action_from_nn, 0)[1]
            #print(action)
            action = action.item()
        if random_for_egreedy < epsilon:
            action = random.sample(set_action, 1)[0]
            #print(action)
        return action

    def optimize(self, no_episodes):
        if len(memory) < batch_size:
            return

        state, action, new_state, reward, done = memory.sample(
            batch_size)  # do something about "done" flag in the enviroemnt and other stuff in order
        state = torch.Tensor(state).to(device)
        new_state = Tensor(new_state).to(device)
        reward = Tensor(reward).to(device)
        action = Tensor(action).to(device)
        done = Tensor(done).to(device)

        ##----------------------------------------Q netwrok and value-----------------------------------------------##

        new_state_Qvalue = self.Qnn(new_state).detach()
        max_new_state_Qvalue = torch.max(new_state_Qvalue, 1)[0]

        target_value_Q = reward + (1 - done) * gamma * max_new_state_Qvalue
        # print(target_value_Q)

        predicted_value_Q = self.Qnn(state).gather(1, action.unsqueeze(1).long()).squeeze(1)
        # print(predicted_value_Q, target_value_Q)
        loss_Q = self.loss_funcQnn(predicted_value_Q, target_value_Q)

        ##---------------------------------------Second moment network and value------------------------------------------------------##

        new_state_SM = self.SMnn(new_state).detach()
        max_new_state_SM = torch.max(new_state_SM, 1)[0]

        target_value_SM = (reward ** 2) + (2 * gamma * reward * max_new_state_Qvalue) + (
                (gamma ** 2) * max_new_state_SM)
        predicted_value_SM = self.SMnn(state).gather(1, action.unsqueeze(1).long()).squeeze(1)

        loss_SM = self.loss_funcSMnn(predicted_value_SM, target_value_SM)

        ##---------------------------------------Updating the Q entwork and second moment network-----------------------------------##
        self.optimizer_Qnn.zero_grad()
        loss_Q.backward()
        self.optimizer_Qnn.step()

        self.optimizer_SMnn.zero_grad()
        loss_SM.backward()
        self.optimizer_SMnn.step()

        ##-------------------------------------- Variance network and value ------------------------------------------------------##

        # new_state_V = self.Vnn(state).detach()
        # max_new_state_V = torch.max(new_state_V,1)[0]

        SM_value = self.SMnn(state).detach().gather(1, action.unsqueeze(1).long()).squeeze(
            1)  # check wether to use updated model or the unupdated

        target_value_v = SM_value - self.Qnn(state).detach().gather(1, action.unsqueeze(1).long()).squeeze(
            1) ** 2  # check wether to use updated model or the unupdated one for Q
        predicted_value_v = self.Vnn(state).gather(1, action.unsqueeze(1).long()).squeeze(1)

        loss_v = self.loss_funcVnn(predicted_value_v, target_value_v)
        self.optimizer_Vnn.zero_grad()
        loss_v.backward()
        self.optimizer_Vnn.step()


def obs_trp_lists(q_array):
    obs_lst = []
    trp_lst = []
    for row in range(10):
        for col in range(10):
            if env.GridWorld["state"][row, col] == "obstacle":
                q_array[row][col] = None
                obs_lst.append((col, row))
            if env.GridWorld["state"][row][col] == "trap":
                q_array[row][col] = None
                trp_lst.append((col, row))

    # list of traps#
    return (obs_lst, trp_lst, q_array)


def Q_array(agent):
    # input tensor with all the state for input

    input = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (1, 0), (1, 1), (1, 2),
             (1, 3), (1, 4), (1, 5),
             (1, 6), (1, 7), (1, 8), (1, 9), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
             (2, 9), (3, 0),
             (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (4, 0), (4, 1), (4, 2), (4, 3),
             (4, 4), (4, 5), (4, 6),
             (4, 7), (4, 8), (4, 9), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9),
             (6, 0), (6, 1),
             (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4),
             (7, 5), (7, 6), (7, 7),
             (7, 8), (7, 9), (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (9, 0),
             (9, 1), (9, 2),
             (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9)]
    input_tensor = torch.Tensor(input).to(device)

    with torch.no_grad():

        Q_array = torch.max(agent.Qnn(input_tensor), 1)[0].cpu().numpy().reshape(10, 10)
        obs_lst, trp_lst, Q_array = obs_trp_lists(q_array=Q_array)
        maxA_array = torch.argmax(agent.Qnn(input_tensor), 1).cpu().numpy().reshape(10, 10)
        fig1, ax1 = plt.subplots(1, 2, figsize=(20, 10))
        heatmap_max = sns.heatmap(Q_array, annot=True, cmap="YlGnBu", ax=ax1[0], fmt=".1f")
        # ax1[0].set_title("epsilon = %f and p = %f" % (round(p, 2), round(q, 2)))
        heatmap_action1 = sns.heatmap(maxA_array, annot=True, cmap="YlGnBu", ax=ax1[1])
        # ax1[1].set_title("0- up , 1 - down, 2 - left, 3 - right")

        for r in obs_lst:
            heatmap_max.add_patch(Rectangle(r, 1, 1, fill=False, edgecolor='green', lw=3))

            heatmap_action1.add_patch(Rectangle(r, 1, 1, fill=False, edgecolor='green', lw=3))

        for s in trp_lst:
            heatmap_max.add_patch(Rectangle(s, 1, 1, fill=True, color="blue", edgecolor='black', lw=3))
            heatmap_max.add_patch(Rectangle((9, 9), 1, 1, fill=False, edgecolor='red', lw=3))

            heatmap_action1.add_patch(Rectangle(s, 1, 1, fill=True, color="blue", edgecolor='black', lw=3))
            heatmap_action1.add_patch(Rectangle((9, 9), 1, 1, fill=False, edgecolor='red', lw=3))
            heatmap_action1.add_patch(Rectangle(s, 1, 1, fill=True, color="blue", edgecolor='black', lw=3))
            heatmap_action1.add_patch(Rectangle((9, 9), 1, 1, fill=False, edgecolor='red', lw=3))
        fig1.show()


if __name__ == "__main__":
    S = "safe"
    env = Gridenv.Grid()

    memory = ExperienceReplay(replay_mem_size)
    agent = Agent()

    for i_episode in range(no_episodes):
        #print("---------------/n")
        state = env.reset()
        # print(state)
        reward = []
        states = []

        while True:
            # print(step)
            action = agent.action_selection(state, epsilon1)
            list_next_state = env.step(action, epsilon2, expert_call)
            terminal_state = list_next_state[-1]
            # if terminal_state:
            # print("terminal_state reached:", state)
            # print(state, action, list_next_state[0], list_next_state[2], list_next_state[-1])
            memory.push(state, action, list_next_state[0], list_next_state[2], list_next_state[-1])
            reward.append(list_next_state[2])
            states.append((state, action))
            agent.optimize(i_episode)

            state = list_next_state[0]

            if terminal_state:
                if sum(reward) == 0.00 or sum(reward) == 100.00:
                    print(states)
                    print(reward)
                    print('retrun in %i episode is %.2f' % (i_episode, sum(reward)))
                    #print(memory.memory)
                # print("steps taken in %i episode are %i" %(i_episode,step))
                # if state == (9,9):
                # print("GOAL")
                if i_episode % 30 == 0:
                    print('retrun in %i episode is %.2f' % (i_episode, sum(reward)))
                # print("hence_exiting:", action)
                break
    Q_array(agent)
# a = torch.randn(1, 2).to(device)
# action = agent.action_selection(a, 0.4)
# a.unsqueeze(0)
# print(action)
# print(agent(a))
