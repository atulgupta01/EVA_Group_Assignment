#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Wed May 27 21:45:37 2020

@author: AtulHome
"""

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque

import cv2 as cv
#from google.colab.patches import cv2_imshow
from PIL import Image
from scipy import ndimage
import copy
from PIL import Image as PILImage
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Initialize the Experience Replay memory
# state[i] = [cropped image state] + [distance to target state]

class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_stateImgs,batch_stateValues, batch_next_stateImgs,batch_next_stateValues = [],[],[],[]
    batch_actions, batch_rewards, batch_dones = [],[],[]
    for i in ind: 
      state, next_state, action, reward, done = self.storage[i]

      batch_stateImgs.append(np.array(state[0],copy=False))
      batch_stateValues.append(np.array(state[1:], copy=False))

      batch_next_stateImgs.append(np.array(next_state[0], copy=False))
      batch_next_stateValues.append(np.array(next_state[1:], copy=False))

      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_stateImgs),np.array(batch_stateValues), np.array(batch_next_stateImgs),np.array(batch_next_stateValues),\
     np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)


# Step 2: Define CNN network for the Actor model
# Same network is to be used for the Actor target

class Actor(nn.Module):

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        ):
        super(Actor, self).__init__()

    # input state image size will 40*40*1

        self.convblock1 = nn.Sequential(nn.Conv2d(in_channels=1,
                out_channels=8, kernel_size=(3, 3), padding=0,
                bias=False), nn.BatchNorm2d(8), nn.ReLU(),
                nn.Dropout(0.1))  # output_size = 38

        self.convblock2 = nn.Sequential(nn.Conv2d(in_channels=8,
                out_channels=16, kernel_size=(3, 3), padding=0,
                bias=False), nn.BatchNorm2d(16), nn.ReLU(),
                nn.Dropout(0.1))  # output_size = 36

        self.convblock3 = nn.Sequential(nn.Conv2d(in_channels=16,
                out_channels=16, kernel_size=(3, 3), padding=0,
                bias=False), nn.BatchNorm2d(16), nn.ReLU(),
                nn.Dropout(0.1))  # putput_size = 34

        self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 17
        self.convblock4 = nn.Sequential(nn.Conv2d(in_channels=16,
                out_channels=8, kernel_size=(1, 1), padding=0,
                bias=False), nn.BatchNorm2d(8), nn.ReLU(),
                nn.Dropout(0.1))  # output_size = 17

        self.convblock5 = nn.Sequential(nn.Conv2d(in_channels=8,
                out_channels=16, kernel_size=(3, 3), padding=0,
                bias=False), nn.BatchNorm2d(16), nn.ReLU(),
                nn.Dropout(0.1))  # output_size = 15

        self.convblock6 = nn.Sequential(nn.Conv2d(in_channels=16,
                out_channels=32, kernel_size=(3, 3), padding=0,
                bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Dropout(0.1))  # output_size = 13

        self.convblock7 = nn.Sequential(nn.Conv2d(in_channels=32,
                out_channels=32, kernel_size=(3, 3), padding=0,
                bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Dropout(0.1))  # output_size = 11

        self.GAP = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))

    # GAP to return 32 and rest of state values to be added before passing to fully connected layer

        self.fc1 = nn.Linear(state_dim - 1 + 32, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state_img, state_val):

        x = self.convblock1(state_img)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.GAP(x)
        x = x.view(-1, 32)

    # concatenate with rest of the state elements

        x = torch.cat([x, state_val], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.max_action * torch.tanh(self.fc3(x))
        return x


# Step 3: Define neural networks for the two Critic models and Critic targets
# Model will be same for all 4 critics

class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

    # Defining the first Critic CNN based network

        self.convblock1_1 = nn.Sequential(nn.Conv2d(in_channels=1,
                out_channels=8, kernel_size=(3, 3), padding=0,
                bias=False), nn.BatchNorm2d(8), nn.ReLU(),
                nn.Dropout(0.1))  # output_size = 38

        self.convblock1_2 = nn.Sequential(nn.Conv2d(in_channels=8,
                out_channels=16, kernel_size=(3, 3), padding=0,
                bias=False), nn.BatchNorm2d(16), nn.ReLU(),
                nn.Dropout(0.1))  # output_size = 36

        self.convblock1_3 = nn.Sequential(nn.Conv2d(in_channels=16,
                out_channels=16, kernel_size=(3, 3), padding=0,
                bias=False), nn.BatchNorm2d(16), nn.ReLU(),
                nn.Dropout(0.1))  # putput_size = 34

        self.pool1_1 = nn.MaxPool2d(2, 2)  # output_size = 17
        self.convblock1_4 = nn.Sequential(nn.Conv2d(in_channels=16,
                out_channels=8, kernel_size=(1, 1), padding=0,
                bias=False), nn.BatchNorm2d(8), nn.ReLU(),
                nn.Dropout(0.1))  # output_size = 17

        self.convblock1_5 = nn.Sequential(nn.Conv2d(in_channels=8,
                out_channels=16, kernel_size=(3, 3), padding=0,
                bias=False), nn.BatchNorm2d(16), nn.ReLU(),
                nn.Dropout(0.1))  # output_size = 15

        self.convblock1_6 = nn.Sequential(nn.Conv2d(in_channels=16,
                out_channels=32, kernel_size=(3, 3), padding=0,
                bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Dropout(0.1))  # output_size = 13

        self.convblock1_7 = nn.Sequential(nn.Conv2d(in_channels=32,
                out_channels=32, kernel_size=(3, 3), padding=0,
                bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Dropout(0.1))  # output_size = 11

        self.GAP1_1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))

    # GAP to return 32 and rest of state values to be added before passing to fully connected layer

        self.fc1_1 = nn.Linear(state_dim - 1 + 32 + action_dim, 400)
        self.fc1_2 = nn.Linear(400, 300)
        self.fc1_3 = nn.Linear(300, 1)

    # Defining the second Critic neural network

        self.convblock2_1 = nn.Sequential(nn.Conv2d(in_channels=1,
                out_channels=8, kernel_size=(3, 3), padding=0,
                bias=False), nn.BatchNorm2d(8), nn.ReLU(),
                nn.Dropout(0.1))  # output_size = 38

        self.convblock2_2 = nn.Sequential(nn.Conv2d(in_channels=8,
                out_channels=16, kernel_size=(3, 3), padding=0,
                bias=False), nn.BatchNorm2d(16), nn.ReLU(),
                nn.Dropout(0.1))  # output_size = 36

        self.convblock2_3 = nn.Sequential(nn.Conv2d(in_channels=16,
                out_channels=16, kernel_size=(3, 3), padding=0,
                bias=False), nn.BatchNorm2d(16), nn.ReLU(),
                nn.Dropout(0.1))  # putput_size = 34

        self.pool2_1 = nn.MaxPool2d(2, 2)  # output_size = 17
        self.convblock2_4 = nn.Sequential(nn.Conv2d(in_channels=16,
                out_channels=8, kernel_size=(1, 1), padding=0,
                bias=False), nn.BatchNorm2d(8), nn.ReLU(),
                nn.Dropout(0.1))  # output_size = 17

        self.convblock2_5 = nn.Sequential(nn.Conv2d(in_channels=8,
                out_channels=16, kernel_size=(3, 3), padding=0,
                bias=False), nn.BatchNorm2d(16), nn.ReLU(),
                nn.Dropout(0.1))  # output_size = 15

        self.convblock2_6 = nn.Sequential(nn.Conv2d(in_channels=16,
                out_channels=32, kernel_size=(3, 3), padding=0,
                bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Dropout(0.1))  # output_size = 13

        self.convblock2_7 = nn.Sequential(nn.Conv2d(in_channels=32,
                out_channels=32, kernel_size=(3, 3), padding=0,
                bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Dropout(0.1))  # output_size = 11

        self.GAP2_1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))

    # we will have 32 values coming from GAP layer and (state_dim-1) other state values

        self.fc2_1 = nn.Linear(state_dim - 1 + 32 + action_dim, 400)
        self.fc2_2 = nn.Linear(400, 300)
        self.fc2_3 = nn.Linear(300, 1)

    def forward(self, state_img, state_val, action):

    # first state element is cropped image
    # Forward-Propagation of the first Critic Network

        x1 = self.convblock1_1(state_img)
        x1 = self.convblock1_2(x1)
        x1 = self.convblock1_3(x1)
        x1 = self.pool1_1(x1)
        x1 = self.convblock1_4(x1)
        x1 = self.convblock1_5(x1)
        x1 = self.convblock1_6(x1)
        x1 = self.convblock1_7(x1)
        x1 = self.GAP1_1(x1)
        x1 = x1.view(-1, 32)

    # concatenate with rest of the state elements

        x1 = torch.cat([x1, state_val, action], 1)
        x1 = F.relu(self.fc1_1(x1))
        x1 = F.relu(self.fc1_2(x1))
        x1 = self.fc1_3(x1)

    # Forward-Propagation of the second Critic Network

        x2 = self.convblock2_1(state_img)
        x2 = self.convblock2_2(x2)
        x2 = self.convblock2_3(x2)
        x2 = self.pool2_1(x2)
        x2 = self.convblock2_4(x2)
        x2 = self.convblock2_5(x2)
        x2 = self.convblock2_6(x2)
        x2 = self.convblock2_7(x2)
        x2 = self.GAP2_1(x2)
        x2 = x2.view(-1, 32)

    # concatenate with rest of the state elements

        x2 = torch.cat([x2, state_val, action], 1)
        x2 = F.relu(self.fc2_1(x2))
        x2 = F.relu(self.fc2_2(x2))
        x2 = self.fc2_3(x2)

        return (x1, x2)

    def Q1(self, state_img, state_val, action):

    # Forward-Propagation of the first Critic Network

        x1 = self.convblock1_1(state_img)
        x1 = self.convblock1_2(x1)
        x1 = self.convblock1_3(x1)
        x1 = self.pool1_1(x1)
        x1 = self.convblock1_4(x1)
        x1 = self.convblock1_5(x1)
        x1 = self.convblock1_6(x1)
        x1 = self.convblock1_7(x1)
        x1 = self.GAP1_1(x1)
        x1 = x1.view(-1, 32)

    # concatenate with rest of the state elements

        x1 = torch.cat([x1, state_val, action], 1)
        x1 = F.relu(self.fc1_1(x1))
        x1 = F.relu(self.fc1_2(x1))
        x1 = self.fc1_3(x1)

        return x1


# Defining new class for step 4 to step 15
# complete training Process is in the given class

class TD3(object):

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        ):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim,
                                  max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = \
            torch.optim.Adam(self.critic.parameters())
        self.max_action = max_action

    def select_action(self, state_img, state_val):
        #state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state_img, state_val).cpu().data.numpy().flatten()

    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=100,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        ):

        for it in range(iterations):

      # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory

            (batch_states, batch_next_states, batch_actions,
             batch_rewards, batch_dones) = \
                replay_buffer.sample(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

      # Step 5: From the next state s’, the Actor target plays the next action a’

            next_action = self.actor_target(next_state)

      # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment

            noise = torch.Tensor(batch_actions).data.normal_(0,
                    policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action,
                    self.max_action)

      # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs

            (target_Q1, target_Q2) = self.critic_target(next_state,
                    next_action)

      # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)

            target_Q = torch.min(target_Q1, target_Q2)

      # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor

            target_Q = reward + ((1 - done) * discount
                                 * target_Q).detach()

      # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs

            (current_Q1, current_Q2) = self.critic(state, action)

      # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)

            critic_loss = F.mse_loss(current_Q1, target_Q) \
                + F.mse_loss(current_Q2, target_Q)

      # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

      # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model

            if it % policy_freq == 0:
                actor_loss = -self.critic.Q1(state,
                        self.actor(state)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

        # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging

                for (param, target_param) in \
                    zip(self.actor.parameters(),
                        self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1
                            - tau) * target_param.data)

        # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging

                for (param, target_param) in \
                    zip(self.critic.parameters(),
                        self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1
                            - tau) * target_param.data)

  # Making a save method to save a trained model

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth'
                   % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth'
                   % (directory, filename))

  # Making a load method to load a pre-trained model

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth'
                                   % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth'
                                    % (directory, filename)))
