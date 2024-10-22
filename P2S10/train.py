# -*- coding: utf-8 -*-
"""
Created on Fri May 29 13:35:08 2020

@author: AtulHome
"""

# -*- coding: utf-8 -*-
"""P2S10.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15iNX6M2kI635N--P07UIb4P-zTqmsFTM
"""

#from google.colab.patches import cv2_imshow
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random, os, time


import ai, map

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

car_file = './images/car.jpg'
city_file = './images/citymap.png'
city_map_file = "./images/MASK1.png"
car_img = cv.imread(car_file)
car1 = map.car(0,0,0)
city1 = map.city(city_file)
citymap1 = map.city(city_map_file)

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def evaluate_policy(policy, eval_episodes=10):
  avg_reward = 0
  for _ in range(eval_episodes):
    obs = env.reset()
    done = False
    print("avg_reward = ", avg_reward)
    while not done:
      stateImg=np.expand_dims(obs[0],1)
      stateImg = torch.Tensor(stateImg).to(device)
      stateValues = np.array(obs[1:], dtype=np.float)
      stateValues = np.expand_dims(stateValues,0)
      stateValues = torch.Tensor(stateValues.reshape(1, -1)).to(device)
      action = policy.select_action(stateImg, stateValues)
      obs, reward, done = env.step(action)
      avg_reward += reward
      #print("avg_reward = ", avg_reward)
  avg_reward /= eval_episodes
  print ("---------------------------------------")
  print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
  print ("---------------------------------------")
  return avg_reward

"""**Function to evaluates the policy by calculating its average reward over 10 episodes**

**Set required parameters**
"""

env = map.env(car1, city1, citymap1, car_img) # Instantiate the environment
seed = 20 # Random seed number
start_timesteps = 1e4 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 5e3 # How often the evaluation step is performed (after how many timesteps)
max_timesteps = 5e5 # Total number of iterations/timesteps
save_models = True # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
batch_size = 100 # Size of the batch
discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005 # Target network update rate
policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated

"""**Define file name for the two saved models: the Actor and Critic models**"""

file_name = "%s_%s" % ("TD3_car", str(seed))
print ("---------------------------------------")
print ("Settings: %s" % (file_name))
print ("---------------------------------------")

"""**Create a folder to store the trained models**"""

if not os.path.exists("./results"):
  os.makedirs("./results")
if save_models and not os.path.exists("./pytorch_models"):
  os.makedirs("./pytorch_models")

"""**Set seeds and get necessary information on the states and actions**"""

torch.manual_seed(seed)
np.random.seed(seed)
state_dim = 2
action_dim = 1
max_action = env.max_action

env.reset()
action = 30
new_obs, reward, done = env.step(action)
print(type(new_obs), reward, done)

policy = ai.TD3(state_dim, action_dim, max_action)
replay_buffer = ai.ReplayBuffer()
#evaluations = [evaluate_policy(policy)]

work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')
max_episode_steps = env._max_episode_steps

total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
t0 = time.time()

# We start the main loop over 500,000 timesteps
while total_timesteps < max_timesteps:
  
  # If the episode is done
  if done:

    # If we are not at the very beginning, we start the training process of the model
    if total_timesteps != 0:
      print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
      policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

    # We evaluate the episode and we save the policy
    if timesteps_since_eval >= eval_freq:
      timesteps_since_eval %= eval_freq
      evaluations.append(evaluate_policy(policy))
      policy.save(file_name, directory="./pytorch_models")
      np.save("./results/%s" % (file_name), evaluations)
    
    # When the training step is done, we reset the state of the environment
    obs = env.reset()
    
    # Set the Done to False
    done = False
    
    # Set rewards and episode timesteps to zero
    episode_reward = 0
    episode_timesteps = 0
    episode_num += 1
  
  # Before 10000 timesteps, we play random actions
  if total_timesteps < start_timesteps:
    action = env.sample_action()
  else: # After 10000 timesteps, we switch to the model
      stateImg=np.expand_dims(obs[0],1)
      stateImg = torch.Tensor(stateImg).to(device)
      stateValues = np.array(obs[1:], dtype=np.float)
      stateValues = np.expand_dims(stateValues,0)
      stateValues = torch.Tensor(stateValues.reshape(1, -1)).to(device)    
      action = policy.select_action(stateImg, stateValues)
      # If the explore_noise parameter is not 0, we add noise to the action and we clip it
      if expl_noise != 0:
          action = (action + np.random.normal(0, expl_noise, size=1)).clip(-max_action, max_action)
  
  # The agent performs the action in the environment, then reaches the next state and receives the reward
  new_obs, reward, done = env.step(action)
  if (total_timesteps%100 == 0):
      print("training", total_timesteps, max_timesteps, episode_reward)
  # We check if the episode is done
  done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
  
  # We increase the total reward
  episode_reward += reward
  
  # We store the new transition into the Experience Replay memory (ReplayBuffer)
  replay_buffer.add((obs, new_obs, action, reward, done_bool))

  # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
  obs = new_obs
  episode_timesteps += 1
  total_timesteps += 1
  timesteps_since_eval += 1

# We add the last policy evaluation to our list of evaluations and we save our model
evaluations.append(evaluate_policy(policy))
if save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
np.save("./results/%s" % (file_name), evaluations)
