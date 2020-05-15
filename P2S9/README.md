# Implementing Twin Delayed Deep Deterministic Policy Gradient (TD3) Algorithm

## **Phase 2 Session 9: Assignment**
---
1. Well, there is a reason why this code is in the image, and not pasted.
2. You need to:
    1. write this code down on a Colab file, upload it to GitHub.
    2. write a Readme file explaining all the 15 steps we have taken:
        1. read me must explain each part of the code
        2. each part of the code must be accompanied with a drawing/image (you cannot use the images from the course content)
    3. Upload the link.
---

## Group Members

Atul Gupta (samatul@gmail.com)

Gaurav Patel (gaurav4664@gmail.com)

Ashutosh Panda (ashusai.panda@gmail.com)

---
## **Twin Delayed DDPG (TD3)**
**TD3** is Q-learning based RL algorithm for environments with continuous action spaces.Original paper can be found ![here](https://arxiv.org/pdf/1802.09477.pdf). This repository implements TD3 algorithm for PyBullet Ant (AntBulletEnv-v0) environment. Entire code can be found [here](). Implementation steps are explained below.

---

### Initializations:
Import all the necessary packages and libraries.

    import os
    import time
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    import pybullet_envs
    import gym
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from gym import wrappers
    from torch.autograd import Variable
    from collections import deque



### Step 1: Define ReplayBuffer class

![Step1_ReplayBuffer](https://github.com/atulgupta01/EVA_Group_Assignment/blob/master/P2S9/Figures/Step1_ReplayBuffer.png)


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
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for i in ind: 
          state, next_state, action, reward, done = self.storage[i]
          batch_states.append(np.array(state, copy=False))
          batch_next_states.append(np.array(next_state, copy=False))
          batch_actions.append(np.array(action, copy=False))
          batch_rewards.append(np.array(reward, copy=False))
          batch_dones.append(np.array(done, copy=False))
        return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)

Object of this class will be used to store the experiences of the agent while acting in the environment. It stores each experience as a tuple **<state,next_state,action,reward,done>**. Its size will be specified as **max_size** at the time of its instance creation.

To utilize this class following methods are defined.

1. add(transition)  
Adds a transition (an experience tuple) into ReplayBuffer. If the memory is full it will add the transition to the beginning. Effectively it implements a circular memory buffer.

2. sample(batch_size)  
It creates a batches of **state,next_state,action,reward and done** of size **batch_size** by uniformly sampling the ReplayBuffer.
These batches are returned as numpy arrays as seen in the last code statement.

This step can be summarised as shown below.


### Step 2: Define Actor class
![Step2_Actor ](https://github.com/atulgupta01/EVA_Group_Assignment/blob/master/P2S9/Figures/Step2_Actor.png)
 

    class Actor(nn.Module):
      def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action
    
      def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x


This class defines the model or network used by Actors. In TD3 algorithm we need two actors, actor_model and actor_target. Since both the actors need to be identical we define a common Actor class. We will use two instances of this class as actor model and actor target.  
We have two methods in this class.

1. \_\_init__(self, state_dims, action_dim, max_action)  
In this method we define the layers used by Actor class. These layers will be connected together later in the **forward()** method to create a network.  
We define 3 layers. Each of the layers are linear layers meaning they are fully connected layers.

2.  forward(self,x)  
This method links the layers defined in the above method to create a complete network. The network takes *state_dim* number of inputs and gives *action_dim* number of outputs. The first two layers have *relu* as its activation functions. The last layer has *tanh* as activation function. It must be noted the output of final layer is multiplied by *max_action*, since output of *tanh* varies from -1 to 1, output of Actor network will vary from *-max_action* to *max_action*.


    

### Step 3: Define Critic class
![Step3_Critic](https://github.com/atulgupta01/EVA_Group_Assignment/blob/master/P2S9/Figures/Step3_Critic.png)
 
    
    class Critic(nn.Module):
      
      def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Defining the first Critic neural network
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)
        # Defining the second Critic neural network
        self.layer_4 = nn.Linear(state_dim + action_dim, 400)
        self.layer_5 = nn.Linear(400, 300)
        self.layer_6 = nn.Linear(300, 1)
    
      def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        # Forward-Propagation on the first Critic Neural Network
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        # Forward-Propagation on the second Critic Neural Network
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2
    
      def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1


This class defines the model or network used by Critics. In TD3 algorithm we need two pairs of competing critics each pair consisting of a critic model  and a critic target. All of these 4 critics need to be identical. Here in this class we jointly define two competing critics. Although they are defined together they both act independently. We will use two instances of this class as critic model and critic target
We have 3 methods in this class.
1. \_\_init__(self, state_dims, action_dim)  
In this method we define the layers used by Critic class. These layers will be connected together later in the **forward()** method to create a network.  
We define 3 layers. Each of the layers are linear layers meaning they are fully connected layers.

2.  forward(self,x)  
This method links the layers defined in the above method to create a complete network. The network concatenates the state and action to generate an input of size *(state_dim + action_dim)*. It generates a single state value as its output. First two layers have *relu* as its activation functions while the last layer has no activation function.  
3. Q1(self, x, u)
This method is similar to *forward()* method except that here forward pass of only the first critic has been defined. This is in order to save the computation in cases where output of only first critic is required hence we need not compute forward pass of both the critics. This method will be useful in **Step 13** while updating the actor model. 



### Step 4-15: Define TD3 class
![Step4_15_TD3](https://github.com/atulgupta01/EVA_Group_Assignment/blob/master/P2S9/Figures/Step4_15_TD3.png)


    class TD3(object):
      
      def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.max_action = max_action
    
      def select_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
        
This class will define our entire TD3 algorithm. It will define all the Actor, Critic objects and its optimizers. It will also define the training algorithm. Two methods of the class are shown in above image.

1. \_\_init__(self, state_dim, action_dim, max_action)  
In this method we define all the objects of TD3 algorithm  
* actor (Acts as Actor Model)
* actor_target (Acts as Actor Target)
* actor_optimizer (Adam optimizer for training Actor model)
* critic (Acts as Critic Model. It has 2 competing Critics inbuilt)
* critic_target (Acts as Critic Target. It has 2 competing Critics inbuilt)
* critic_optimizer (Adam optimizer for training Critic model)

  We also copy Actor Model into Actor Target (This is done to have identical model and targets at the beginning)  

        self.actor_target.load_state_dict(self.actor.state_dict())
  Similarly we copy Critic Model into Critic Target  

        self.critic_target.load_state_dict(self.critic.state_dict())
 2. select_action(self, state)  
 In this method we find the best action for given state.  
 For this  we first reshape input *state* into a row vector then convert it to a pytorch tensor because the torch model expects a tensor. As shown below.  
 
        state = torch.Tensor(state.reshape(1, -1)).to(device)  
        
    Here *to(device)* transfers state to a cpu or gpu as per the availability.  
    Next, the *state* tensor is forward passed through the Actor Model network to get the computed *action*. The action is converted to numpy and flattened (make 1D) before returning.
### Step 4: Define Train Method of TD3 class 

![Step4_Sample](https://github.com/atulgupta01/EVA_Group_Assignment/blob/master/P2S9/Figures/Step4_Sample.png)

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    
    for it in range(iterations):
      
      # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
      batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
      state = torch.Tensor(batch_states).to(device)
      next_state = torch.Tensor(batch_next_states).to(device)
      action = torch.Tensor(batch_actions).to(device)
      reward = torch.Tensor(batch_rewards).to(device)
      done = torch.Tensor(batch_dones).to(device)

For each of the iterations, sample a batch of size *batch_size* from ReplayBuffer.  

    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
  If we refer to *sample()* method of ReplayBuffer class, we were returning batch components as numpy_arrays while the pytorch networks expect inputs as tensors hence we convert all of the sampled components into pytorch tensors.  
  
    state = torch.Tensor(batch_states).to(device)
    next_state = torch.Tensor(batch_next_states).to(device)
    action = torch.Tensor(batch_actions).to(device)
    reward = torch.Tensor(batch_rewards).to(device)
    done = torch.Tensor(batch_dones).to(device)

### Step 5: Action Target Forward Pass

![Step5_ActorTargetForward](https://github.com/atulgupta01/EVA_Group_Assignment/blob/master/P2S9/Figures/Step5.png)
 

Generate actions for next_states through Actor Target.  

    next_action = self.actor_target(next_state)
It must be noted here that *next_state* here corresponds to entire batch of next states obtained by sampling the ReplayBuffer in the previous step. Also we are using the Actor Target and not Actor Model. In TD3 algorithm for acting purpose Actor Model is used while for equation calculations Actor Model is used. *next_action* obtained here will be used in calculation of Q-Values.

### Step 6: Add noise to predicted actions
![Step6_AddNoise](https://github.com/atulgupta01/EVA_Group_Assignment/blob/master/P2S9/Figures/Step6_AddNoise.png)


    noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
    noise = noise.clamp(-noise_clip, noise_clip)
    next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
    
Noise values are sampled from a normal distribution with 0 mean and *policy_noise* as standard deviation. *noise* here is a tensor of size same as *batch_actions* i.e. batch_size .

    noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
We don't want noise values to be too high hence we clip the noise value.  

    noise = noise.clamp(-noise_clip, noise_clip)
Finally we add the noise to *next_action*. We again clip the values to range of allowed action values.

    next_action = (next_action + noise).clamp(-self.max_action, self.max_action)



### Step 7: Critic Target Forward pass
![Step7_CriticTarget](https://github.com/atulgupta01/EVA_Group_Assignment/blob/master/P2S9/Figures/Step7_CriticTarget.png)

Obtain Q-Values from Critic Targets for sampled batch of *next_state* and *next_actions* generated by Actor Target.  

    target_Q1, target_Q2 = self.critic_target(next_state, next_action)
It must again be noted that we are using Critic Target here since these Q-values will be used in equation computations. When we are acting in the environment we will used Critic Model. 

### Step 8: Minimum of Critic Target Values
![Step8_CriticMin](https://github.com/atulgupta01/EVA_Group_Assignment/blob/master/P2S9/Figures/Step8_CriticMin.png)

Select the minimum of two Critic Q-values.

    target_Q = torch.min(target_Q1, target_Q2)

### Step 9: Compute Expected Q-Values

![Step9_CriticBelmanEqn](https://github.com/atulgupta01/EVA_Group_Assignment/blob/master/P2S9/Figures/Step9_CriticBelmanEqn.png)

Compute the expected Q value for current state using the Bellman Equation. This value will be compared with predicted Q-Value later to get the temporal difference. 

    target_Q = reward + ((1 - done) * discount * target_Q).detach()
Some important points.  

* Multiplication by *(1 - done)* :  
This is to take care of the terminal states (episode ends).  
So whenever the transition is a terminal state, we will have *done=1* hence *(1-done)=0* so discount part will be made 0. This is because in this case *next_state*,*next_action* and *target_Q* will be meaningless.  
Whenever transition is non-terminal state, *target_Q* will be normally evaluated because *done=0*

* *detach()* 
This is to detach *target_Q* computation from the computation graph. Doing this, pytorch will not track the operations on this subgraph.

### Step 10: Compute Predicted Q-Values
![Step10_CriticModel](https://github.com/atulgupta01/EVA_Group_Assignment/blob/master/P2S9/Figures/Step10_CriticModel.png)

Obtain predicted Q-Values from Model Critics for current state and action.  

    current_Q1, current_Q2 = self.critic(state, action)
    
Here we obtain Q-Values from two critics for a batch *current_state* and *current_action* obtained through ReplayBuffer sampling. It must be noted here that we are using Critic Model and not Critic Target. 


### Step 11: Compute Critic loss
![Step11_CriticLoss](https://github.com/atulgupta01/EVA_Group_Assignment/blob/master/P2S9/Figures/Step11_CriticLoss.png)
 

    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
We have already calculated expected Q-Values (*target_Q*) and predicted Q-Values (*current_Q1, current_Q2*). We find the temporal difference between them to get Critic loss. 

It must be noted that we have calculated combined loss of Critic 1 and Critic 2, this is because both the critics are defined in Critic class together. So *critic_loss* will be used for combined training of the critics.

### Step 12: Update Critic Models
![Step12_CriticModelUpdate](https://github.com/atulgupta01/EVA_Group_Assignment/blob/master/P2S9/Figures/Step12_CriticModelUpdate.png)

Update Critic Model through back propagation.  

    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()
    
We first make gradients of *critic_optimizer* zero. Then we propagate *critic_loss* backwards so that gradients are recorded. Finally we ask *critic_optimizer to update Critic Model. 

It must be noted here that we are updating Critic Model. Critic Targets are never updated through backpropagation. They are updated throgh Prolyak Averaging.



### Step 13: Update Actor Model
![Step13_ActorModelUpdate](https://github.com/atulgupta01/EVA_Group_Assignment/blob/master/P2S9/Figures/Step13_ActorModelUpdate.png)

Update Actor Model through backpropagation every *policy_freq*'th iteration.

    if it % policy_freq == 0:
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

We first find whether this iteration is divisible by *policy_freq*
    
    if it % policy_freq == 0:
    
We calculate *actor_loss*. We need to update Actor Model parameters such that action taken by it maximizes Q-values output of Critic Model. So we select *actor_loss* as follows. 

    actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
Since we have -ve sign, optimizer will perform gradient ascent on Actor Model parameter so as to maximize Critic Q-Value.  
*mean()* here corresponds to average of *actor_loss* over a batch.

Next we initialize actor_optimizer gradients to zero, perform backward pass to record gradients and finally step through to update Actor Model parameters.
    
### Step 14: Update Actor Target
![Step14_ActorTargetUpdate](https://github.com/atulgupta01/EVA_Group_Assignment/blob/master/P2S9/Figures/Step14_ActorTargetUpdate.png)

Update Actor Target through Polyak Averaging.

    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
          
We generate an iterator of matched pair of Actor Model and Actor Target parameters by zipping them as

    zip(self.actor.parameters(), self.actor_target.parameters())
    
We now loop through this iterator and copy Actor Model parameters into Actor Target with Polyak averaging.

### Step 15: Update Critic Target
![Step15_CriticTargetUpdate](https://github.com/atulgupta01/EVA_Group_Assignment/blob/master/P2S9/Figures/Step15_CriticTargetUpdate.png)

Update Critic Target through Polyak Averaging.

    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
          
We generate an iterator of matched pair of Critic Model and Critic Target parameters by zipping them as

    zip(self.critic.parameters(), self.critic_target.parameters())
    
We now loop through this iterator and copy Critic Model parameters into Critic Target with Polyak averaging.


    
