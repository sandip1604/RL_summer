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
import time
import random
import numpy as np
import os.path

plt.style.use('ggplot')

use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)
Tensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

env = Gridenv.Grid()

##### Params #########################

learning_rate = 0.005
gamma = 0.9
batch_size = 150
replay_mem_size = 50000

epsilon1 = 0.1
epsilon2 = 0.3

hidden_layer1 = 10 #20
hidden_layer2 = 32 #50
hidden_layer3 = 64 #200
hidden_layer4 = 20  #50
#hidden_layer5 = 20  #20

SM_hidden_layer1 = 20
SM_hidden_layer2 = 60

number_of_output = 4
# epsilon1 = 0.1
# epsilon2 = 0.3
# learning_rate = 0.2
discount_rate = 0.9
no_episodes = 10000
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
        #self.linear4 = nn.Linear(hidden_layer3, hidden_layer4)
        #self.linear5 = nn.Linear(hidden_layer4, hidden_layer5)
        self.linear6 = nn.Linear(hidden_layer3, number_of_output)
        self.dropout = nn.Dropout(p=0)

        # self.value1 = nn.Linear(hidden_layer3, hidden_layer4)
        # self.value3 = nn.Linear(hidden_layer4, 1)


        #self.value2 = nn.Linear(hidden_layer4, hidden_layer5)

        self.activation = nn.ReLU()

        torch.nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.linear3.weight, nonlinearity='relu')
        # torch.nn.init.kaiming_normal_(self.linear4.weight, nonlinearity='relu')
        # torch.nn.init.kaiming_normal_(self.linear5.weight, nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.linear6.weight, nonlinearity='relu')
        # torch.nn.init.kaiming_normal_(self.value1.weight, nonlinearity='relu')
        # torch.nn.init.kaiming_normal_(self.value3.weight, nonlinearity='relu')


    def forward(self, x):
        # code to normalise the input
        x = x / 10

        output_linear = self.dropout(self.activation(self.linear1(x)))
        output_linear = self.dropout(self.activation(self.linear2(output_linear)))
        output_linear1 = self.activation(self.linear3(output_linear))

        #output_linear = self.linear4(self.dropout(output_linear1))
        #output_linear = self.linear5(self.dropout(self.activation(output_linear)))
        output_linear = self.linear6(self.activation(output_linear1))
        # output_linear = F.softmax(output_linear, dim=1)              # included in loss function

        # output_value = self.value1(output_linear1)
        # output_value = self.value3(self.activation(output_value))


        # output_value = self.value2(self.dropout(self.activation(output_value)))


        output_final = output_linear


        return output_final


class NeuralNetwork_V(nn.Module):
    def __init__(self):
        super(NeuralNetwork_V, self).__init__()

        self.linear1 = nn.Linear(2, hidden_layer1)
        self.linear2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.linear3 = nn.Linear(hidden_layer2, hidden_layer3)
        # self.linear4 = nn.Linear(hidden_layer3, hidden_layer4)
        # self.linear5 = nn.Linear(hidden_layer4, hidden_layer5)
        self.linear6 = nn.Linear(hidden_layer3, number_of_output)
        self.dropout = nn.Dropout(p=0)

        self.value1 = nn.Linear(hidden_layer3, hidden_layer4)
        self.value3 = nn.Linear(hidden_layer4, 1)


        # self.value2 = nn.Linear(hidden_layer4, hidden_layer5)

        self.activation = nn.ReLU()

        torch.nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.linear3.weight, nonlinearity='relu')
        # torch.nn.init.kaiming_normal_(self.linear4.weight, nonlinearity='relu')
        # torch.nn.init.kaiming_normal_(self.linear5.weight, nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.linear6.weight, nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.value1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.value3.weight, nonlinearity='relu')

    def forward(self, x):
        # code to normalise the input
        x = x / 10

        output_linear = self.dropout(self.activation(self.linear1(x)))
        output_linear = self.dropout(self.activation(self.linear2(output_linear)))
        output_linear1 = self.activation(self.linear3(output_linear))

        # output_linear = self.linear4(self.dropout(output_linear1))
        # output_linear = self.linear5(self.dropout(self.activation(output_linear)))
        output_linear = self.linear6(self.activation(output_linear1))
        # output_linear = F.softmax(output_linear, dim=1)              # included in loss function

        # output_value = self.value1(output_linear1)
        # output_value = self.value3(self.activation(output_value))


        #output_value = self.value2(self.dropout(self.activation(output_value)))


        output_final = output_linear

        return output_final


class NeuralNetwork_SM(nn.Module):
    def __init__(self):
        super(NeuralNetwork_SM, self).__init__()

        self.linear1 = nn.Linear(2, hidden_layer1)
        self.linear2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.linear3 = nn.Linear(hidden_layer2, hidden_layer3)
        # self.linear4 = nn.Linear(hidden_layer3, hidden_layer4)
        # self.linear5 = nn.Linear(hidden_layer4, hidden_layer5)
        self.linear6 = nn.Linear(hidden_layer3, number_of_output)
        self.dropout = nn.Dropout(p=0)

        # self.value1 = nn.Linear(hidden_layer3, hidden_layer4)
        # self.value3 = nn.Linear(hidden_layer4, 1)


        # self.value2 = nn.Linear(hidden_layer4, hidden_layer5)

        self.activation = nn.ReLU()

        # torch.nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        # torch.nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='relu')
        # torch.nn.init.kaiming_normal_(self.linear3.weight, nonlinearity='relu')
        # # torch.nn.init.kaiming_normal_(self.linear4.weight, nonlinearity='relu')
        # # torch.nn.init.kaiming_normal_(self.linear5.weight, nonlinearity='relu')
        # # torch.nn.init.kaiming_normal_(self.linear6.weight, nonlinearity='relu')
        # torch.nn.init.kaiming_normal_(self.value1.weight, nonlinearity='relu')
        # torch.nn.init.kaiming_normal_(self.value3.weight, nonlinearity='relu')

    def forward(self, x):
        # code to normalise the input
        x = x / 10

        output_linear = self.dropout(self.activation(self.linear1(x)))
        output_linear = self.dropout(self.activation(self.linear2(output_linear)))
        output_linear1 = self.activation(self.linear3(output_linear))

        # output_linear = self.linear4(self.dropout(output_linear1))
        # output_linear = self.linear5(self.dropout(self.activation(output_linear)))
        output_linear = self.linear6(self.activation(output_linear1))
        # output_linear = F.softmax(output_linear, dim=1)              # included in loss function

        # output_value = self.value1(output_linear1)
        # output_value = self.value3(self.activation(output_value))


        #output_value = self.value2(self.dropout(self.activation(output_value)))


        output_final =  output_linear

        return output_final


class Agent(object):
    def __init__(self):
        self.Qnn = NeuralNetwork_Q().to(device)
        self.SMnn = NeuralNetwork_SM().to(device)
        self.Vnn = NeuralNetwork_V().to(device)


        self.loss_funcSMnn = nn.SmoothL1Loss()  # for discrete action space
        self.loss_funcVnn = nn.MSELoss()  # for discrete action space
        self.loss_funcQnn = nn.MSELoss()  # for discrete action space

        self.optimizer_Qnn = optim.Adam(params=self.Qnn.parameters(), lr=learning_rate)
        self.optimizer_SMnn = optim.Adam(params=self.SMnn.parameters(), lr=learning_rate/5)
        self.optimizer_Vnn = optim.Adam(params=self.Vnn.parameters(), lr=learning_rate)

    def action_selection(self, state, epsilon):

        random_for_egreedy = torch.rand(1).item()
        #print(random_for_egreedy)
        set_action = {0, 1, 2, 3}
        # print(state)
        with torch.no_grad():
            state = Tensor(state)
            self.Qnn.eval()
            action_from_nn = self.Qnn(state)
            self.Qnn.train()
            #print(action_from_nn)
            action = torch.max(action_from_nn, 0)[1]
            #print(action.device)
            action = action.item()
        if random_for_egreedy < epsilon:
            action = random.sample(set_action, 1)[0]
            #print(action)
        return action

    def optimize(self, k):
        if len(memory) < batch_size:
            return

        state, action, new_state, reward, done = memory.sample(batch_size)  # do something about "done" flag in the enviroemnt and other stuff in order
        state = Tensor(state)
        new_state = Tensor(new_state)
        reward = Tensor(reward)
        action = Tensor(action)
        done = Tensor(done)

        ##----------------------------------------Q netwrok and value-----------------------------------------------##

        new_state_Qvalue = self.Qnn(new_state).detach()
        max_new_state_Qvalue = torch.max(new_state_Qvalue, 1)[0]

        target_value_Q = reward + (1 - done) * gamma * max_new_state_Qvalue
        # print(target_value_Q)

        predicted_value_Q = self.Qnn(state).gather(1, action.unsqueeze(1).long()).squeeze(1)
        # print(predicted_value_Q, target_value_Q)
        loss_Q = self.loss_funcQnn(predicted_value_Q, target_value_Q)
        #print(loss_Q)

        ##---------------------------------------Second moment network and value------------------------------------------------------##

        new_state_SM = self.SMnn(new_state).detach()
        max_new_state_SM = torch.max(new_state_SM, 1)[0]
        if k % 1000 == 0:
            print(state)
            print(reward)
            print((reward ** 2) + (2 * gamma * reward * max_new_state_Qvalue) + ((gamma ** 2) * max_new_state_SM))
            print(2 * gamma * reward * max_new_state_Qvalue)
            print( max_new_state_SM)
        target_value_SM = (reward ** 2) + (2 * gamma * reward * max_new_state_Qvalue) + ((gamma ** 2) * max_new_state_SM)
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

        # SM_value = self.SMnn(state).detach().gather(1, action.unsqueeze(1).long()).squeeze(1)  # check wether to use updated model or the unupdated
        #
        # target_value_v = SM_value - (self.Qnn(state).detach().gather(1, action.unsqueeze(1).long()).squeeze(1)) ** 2  # check wether to use updated model or the unupdated one for Q
        # predicted_value_v = self.Vnn(state).gather(1, action.unsqueeze(1).long()).squeeze(1)
        #
        # loss_v = self.loss_funcVnn(predicted_value_v, target_value_v)
        # self.optimizer_Vnn.zero_grad()
        # loss_v.backward()
        # self.optimizer_Vnn.step()


def obs_trp_lists(q_array, a_array, var_array):
    obs_lst = []
    trp_lst = []
    for row in range(10):
        for col in range(10):
            if env.GridWorld["state"][row][col] == "obstacle":
                q_array[row][col] = np.nan
                a_array[row][col] = np.nan
                var_array[row][col] = np.nan
                obs_lst.append((col, row))
            if env.GridWorld["state"][row][col] == "trap":
                q_array[row][col] = np.nan
                a_array[row][col] = np.nan
                var_array[row][col] = np.nan
                trp_lst.append((col, row))

    # list of traps#
    return (obs_lst, trp_lst, q_array, a_array, var_array)


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
    input_tensor = Tensor(input)

    with torch.no_grad():
        #agent.Qnn.eval()

        Q_array = torch.max(agent.Qnn(input_tensor), 1)[0].cpu().numpy().reshape(10, 10)
        action_tensor = torch.argmax(agent.Qnn(input_tensor), 1)
        print(action_tensor.size())
        maxA_array = action_tensor.cpu().numpy().astype(float).reshape(10, 10)

        SM_array = agent.SMnn(input_tensor).gather(1, action_tensor.unsqueeze(1).long()).cpu().numpy().reshape(10,10)
        print(SM_array)

        Var_array = SM_array

        obs_lst, trp_lst, Q_array, maxA_array, Var_array = obs_trp_lists(q_array=Q_array, a_array= maxA_array, var_array = Var_array)

        fig1, ax1 = plt.subplots(1, 2, figsize=(20, 10))
        fig2, ax2 = plt.subplots(1, 2, figsize=(20, 10))
        heatmap_Qmax = sns.heatmap(Q_array, annot=True, cmap="YlGnBu", ax=ax1[0], fmt=".1f")
        heatmap_Vmax = sns.heatmap(Var_array, annot=True, cmap="YlGnBu", ax=ax2[0], fmt=".1f")
        # ax1[0].set_title("epsilon = %f and p = %f" % (round(p, 2), round(q, 2)))
        heatmap_action1 = sns.heatmap(maxA_array, annot=True, cmap="YlGnBu", ax=ax1[1])
        heatmap_action2 = sns.heatmap(maxA_array, annot=True, cmap="YlGnBu", ax=ax2[1])
        # ax1[1].set_title("0- up , 1 - down, 2 - left, 3 - right")

        for r in obs_lst:
            heatmap_Qmax.add_patch(Rectangle(r, 1, 1, fill=False, edgecolor='green', lw=3))
            heatmap_Vmax.add_patch(Rectangle(r, 1, 1, fill=False, edgecolor='green', lw=3))
            heatmap_action1.add_patch(Rectangle(r, 1, 1, fill=False, edgecolor='green', lw=3))
            heatmap_action2.add_patch(Rectangle(r, 1, 1, fill=False, edgecolor='green', lw=3))

        for s in trp_lst:
            heatmap_Qmax.add_patch(Rectangle(s, 1, 1, fill=True, color="blue", edgecolor='black', lw=3))
            heatmap_Qmax.add_patch(Rectangle((9, 9), 1, 1, fill=False, edgecolor='red', lw=3))
            heatmap_Vmax.add_patch(Rectangle(s, 1, 1, fill=True, color="blue", edgecolor='black', lw=3))
            heatmap_Vmax.add_patch(Rectangle((9, 9), 1, 1, fill=False, edgecolor='red', lw=3))

            heatmap_action1.add_patch(Rectangle(s, 1, 1, fill=True, color="blue", edgecolor='black', lw=3))
            heatmap_action1.add_patch(Rectangle((9, 9), 1, 1, fill=False, edgecolor='red', lw=3))
            heatmap_action2.add_patch(Rectangle(s, 1, 1, fill=True, color="blue", edgecolor='black', lw=3))
            heatmap_action2.add_patch(Rectangle((9, 9), 1, 1, fill=False, edgecolor='red', lw=3))
        fig1.savefig('fig3.png', bbox="tight", dpi=200)
        plt.close(fig1)
        fig2.savefig('fig4.png', bbox="tight", dpi=200)
        plt.close(fig2)
        #fig1.show()


if __name__ == "__main__":
    S = "safe"
    t1 = time.time()
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

            memory.push(state, action, list_next_state[0], list_next_state[2], list_next_state[-1])
            reward.append(list_next_state[2])
            states.append((state, action))
            agent.Qnn.train()
            agent.optimize(i_episode, i_episode)

            state = list_next_state[0]

            if terminal_state:
                if i_episode % 1000 == 0:
                    print('retrun in %i episode is %.2f' % (i_episode, sum(reward)))
                    Q_array(agent)
                # print("hence_exiting:", action)
                break
    Q_array(agent)
    t2 = time.time()-t1
    print(t2)

