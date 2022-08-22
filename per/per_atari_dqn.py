import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torchvision.datasets as dsets
import numpy as np
from torch.distributions.normal import Normal

import torchvision.transforms as transforms

import collections
import random

import gym

from proportional import *
from ranking import *
from atari_setup import *

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#to use Atari games
#!pip install gym[atari,accept-rom-license]==0.21.0

class Q_Module(nn.Module):
  def __init__(self, a_dim):
    super(Q_Module, self).__init__()
    self.a_dim=a_dim
    #input: 4x84x84 (4 stacked grayscale images)
    self.conv1=nn.Conv2d(4,16,(8,8),stride=4).to(device)
    self.conv2=nn.Conv2d(16,32,(4,4),stride=2).to(device)

    self.fc1=nn.Linear(32*9*9, 256).to(device)
    #sibling layers for 4 actions
    self.output1=nn.Linear(256,1).to(device)
    self.output2=nn.Linear(256,1).to(device)
    self.output3=nn.Linear(256,1).to(device)
    self.output4=nn.Linear(256,1).to(device)
    self.element_init()

  def element_init(self): #initialize and get sizes of parameters
    for element in self.children():
      if isinstance(element, nn.Conv2d):
        nn.init.xavier_uniform_(element.weight)
        nn.init.zeros_(element.bias)
      elif isinstance(element, nn.Linear):
        nn.init.xavier_uniform_(element.weight)
        nn.init.zeros_(element.bias)
    return
    
  def forward(self, pre_processed_obs, a_idx):
    relu=nn.ReLU()
    convs=nn.Sequential(self.conv1, relu, self.conv2, relu)
    features=convs(pre_processed_obs)
    feature_vector=features.reshape(-1)
    if a_idx==0:
      output=self.output1
    elif a_idx==1:
      output=self.output2
    elif a_idx==2:
      output=self.output3
    else:
      output=self.output4
    fc_layers=nn.Sequential(self.fc1, relu, output)
    action_value=fc_layers(feature_vector)
    return action_value

class DQN_Agent():
  def __init__(self, env, test_env):
    self.env=env
    self.test_env=test_env
    
    self.a_dim=self.env.action_space.n
    self.action_list=np.arange(0, self.a_dim, 1)

    self.qm=Q_Module(self.a_dim)
    self.target_qm=Q_Module(self.a_dim)
    #align parameters
    polyak_averaging(self.qm, self.target_qm, polyak=0)

  def get_action(self, proc_obs, eps): #use eps to switch between random(e=1), e-greedy, greedy(e=0)
    if eps==1:
      action=self.env.action_space.sample()
    else:
      #e-greedy as basic
      q_tensor=torch.zeros(self.a_dim).to(device)
      for action in self.action_list:
        q_tensor[action]=self.qm(proc_obs, action)
      argmax_idx=torch.argmax(q_tensor, dim=0)
      probability=[eps/self.a_dim for _ in range(self.a_dim)]
      probability[argmax_idx]+=(1-eps)
      action=np.random.choice(self.action_list, p=probability)
    return action

class Obs_Stack():
  def __init__(self, max_length):
    self.max_length=max_length
    self.data=torch.Tensor([]).to(device)
    self.length=0
  
  def add_item(self, item):
    if self.length==self.max_length:
      self.data=self.data[1:,:,:] #remove first data
      self.length-=1
    self.data=torch.cat((self.data, item), dim=0)
    self.length+=1
    return
  
  def get_stack(self):
    if self.length!=self.max_length: #not full
      return "invalid"
    else: #full
      return self.data
    
Ep_Step=collections.namedtuple("Ep_Step", field_names=['obs', 'action', 'reward', 'obs_f', 'priority', 'termin_signal'])

class Double_DQN(): #train using double dqn
  def __init__(self, agent, stack_count, per):
    self.agent=agent
    if per=='proportional':
      self.replay_buffer=Proportional_Prioritized_Replay()
    else:
      assert per=='ranking', 'invalid per option'
      self.replay_buffer=Ranking_Prioritized_Replay()
    
    self.obs_stack=Obs_Stack(max_length=4)
    self.test_obs_stack=Obs_Stack(max_length=4)
  
  def check_performance(self):
    avg_return=0
    avg_len_ep=0
    for idx in range(1,10+1):
      ep_return=0
      len_ep=1
      obs=self.agent.test_env.reset()
      #fill obs stack
      while self.test_obs_stack.get_stack()=='invalid':
        self.test_obs_stack.add_item(obs)
        len_ep+=1
        obs, reward, _,_=self.agent.test_env.step(self.agent.test_env.action_space.sample())
        ep_return+=reward
      
      proc_obs=self.test_obs_stack.get_stack()
      while True:
        action=self.agent.get_action(proc_obs, eps=0)
        obs_f, reward, termin_signal, _=self.agent.test_env.step(action)
        self.test_obs_stack.add_item(obs_f)
        proc_obs_f=self.test_obs_stack.get_stack()

        ep_return+=reward
        if termin_signal:
          break
        else:
          proc_obs=proc_obs_f
          len_ep+=1
      avg_return=avg_return+(ep_return-avg_return)/idx
      avg_len_ep=avg_len_ep+(len_ep-avg_len_ep)/idx
    return avg_return, avg_len_ep
  
  def get_value_loss(self, gamma, batch_data, isrs):
    batch_size=len(batch_data)
    #apply isr to each sample -> compute error (effect of modifying step-size for each sample)
    value_loss=torch.FloatTensor([0]).to(device)
    for b_idx, sample_idx in enumerate(batch_data):
      isr=isrs[b_idx]
      ep_step=self.replay_buffer.data[sample_idx]
      proc_obs=ep_step.obs
      action=ep_step.action
      reward=ep_step.reward
      proc_obs_f=ep_step.obs_f
      termin_signal=ep_step.termin_signal

      #choose action_f using qm
      action_f=self.agent.get_action(proc_obs, eps=0) #greedy for action selection
      Q_f=self.agent.target_qm(proc_obs_f, action_f)
      update_target=reward+(1-termin_signal)*gamma*Q_f.item()

      Q=self.agent.qm(proc_obs, action)
      tde=update_target-Q
      #apply isr to adjust step size for each sample
      value_loss=value_loss+isr*(tde)**2

      #update tde in samples
      self.replay_buffer.data[sample_idx]=self.replay_buffer.data[sample_idx]._replace(priority=torch.abs(tde).detach())

    value_loss=value_loss/batch_size
    return value_loss
  
  def train(self, gamma, batch_size, n_epochs, steps_per_epoch, start_after, update_after, update_every, action_every, lr, polyak, alpha, beta):
    #set optimizers
    value_optim=optim.SGD(self.agent.qm.parameters(), lr=lr)

    #initialize
    eps=1
    step=1
    update=False
    obs=self.agent.env.reset()

    #fill obs_stack
    while self.obs_stack.get_stack()=='invalid':
      self.obs_stack.add_item(obs)
      #add no-op frames
      obs,_,_,_=self.agent.env.step(0)

    for epoch in range(1, n_epochs+1):
      #epoch start
      while True:
        if step>start_after:
          eps=0.1
        if step>update_after:
          update=True
        
        #data experience
        proc_obs=self.obs_stack.get_stack()

        #choose action every N steps
        if step%action_every==1:
          action=self.agent.get_action(proc_obs, eps)
        obs_f, reward, termin_signal, _=self.agent.env.step(action)
        self.obs_stack.add_item(obs_f)
        proc_obs_f=self.obs_stack.get_stack()

        #get td-error as priority
        Q=self.agent.qm(proc_obs, action)
        action_f=self.agent.get_action(proc_obs_f, eps=0)
        Q_f=self.agent.qm(proc_obs_f, action_f)
        priority=torch.abs(reward+gamma*Q_f-Q).detach()
        #add data
        ep_step=Ep_Step(proc_obs, action, reward, proc_obs_f, priority, termin_signal)
        self.replay_buffer.add_item(ep_step)

        if update and step%update_every==0:
          self.replay_buffer.update(alpha=alpha)
          for u_step in range(update_every):
            batch_data, isrs=self.replay_buffer.sample(batch_size, alpha, beta)
            value_loss=self.get_value_loss(gamma, batch_data, isrs)
            #print(value_loss)

            value_optim.zero_grad()
            value_loss.backward()
            value_optim.step()

          #after update: set target parameters to follow current parameters
          polyak_averaging(self.agent.qm, self.agent.target_qm, polyak)
        
        proc_obs=proc_obs_f
        step+=1
        if step%steps_per_epoch==0:
          break
      #end of epoch:
      avg_return, avg_len_ep=self.check_performance()
      print("Epoch: {:d}, Avg_Return: {:.3f}, Avg_Len_Ep: {:.2f}".format(epoch, avg_return, avg_len_ep))
    return

env=Rew_Wrapper(Obs_Wrapper(gym.make('Breakout-v4')))
test_env=Rew_Wrapper(Obs_Wrapper(gym.make('Breakout-v4')))
dqn_agent=DQN_Agent(env, test_env)

double_dqn=Double_DQN(dqn_agent, stack_count=4, per='proportional')
double_dqn.train(gamma=0.99,  batch_size=64, n_epochs=100, steps_per_epoch=10000, start_after=100000, 
                 update_after=200000, update_every=4, action_every=4, lr=2.5e-4, polyak=.995, alpha=0.5, beta=0.5)