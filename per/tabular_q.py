import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torchvision.datasets as dsets
import numpy as np
from torch.distributions.normal import Normal

import collections
import random

import gym

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from blind_cliffwalk import Blind_Cliffwalk, Observation_Wrapper
from proportional import Proportional_Prioritized_Replay
from ranking import Ranking_Prioritized_Replay

class Tabular_Agent():
  def __init__(self, env, test_env, gamma, lambd):
    self.env=env
    self.test_env=test_env
    self.gamma=gamma
    self.lambd=lambd

    self.o_dim=self.env.observation_space.n
    self.a_dim=self.env.action_space.n

    #self.action_values=torch.zeros(self.o_dim, self.a_dim).to(device)
    self.action_values=torch.normal(torch.zeros(self.o_dim, self.a_dim), torch.full([self.o_dim, self.a_dim], 0.1)).to(device)
  
  def get_action(self, obs, a_mode):
    if a_mode=='random':
      action=self.env.action_space.sample()
    else:
      assert a_mode=='policy', 'invalid action mode'
      q_values=self.action_values[obs, :]
      q_values=q_values.squeeze(dim=0)
      action=torch.argmax(q_values, dim=0).item() #greedy
    return action

class Tabular_Q_Learning():
  def __init__(self, agent, per):
    self.agent=agent
    if per=='proportional':
      self.replay_buffer=Proportional_Prioritized_Replay()
    else:
      self.replay_buffer=Ranking_Prioritized_Replay()
    
  def check_performance(self):
    #average 10 episodes
    avg_return=0
    for idx in range(10):
      ep_return=0
      obs=self.agent.test_env.reset()
      while True:
        action=self.agent.get_action(obs, 'policy')
        obs_f, reward, termin_signal,_=self.agent.test_env.step(action)
        ep_return+=reward
        if termin_signal:
          break
        else:
          obs=obs_f
      avg_return=avg_return+(ep_return-avg_return)/(idx+1)
    
    return avg_return
  
  def train(self, batch_size, n_epochs, steps_per_epoch, start_after, update_after, update_every, lr, alpha, beta):
    a_mode='random'
    update=False
    obs=self.agent.env.reset()
    step=1

    for epoch in range(1,n_epochs+1):
      while True:
        #for each step
        if step>start_after:
          a_mode='policy'
        if step>update_after:
          update=True
        
        #take a step in the env.
        action=self.agent.get_action(obs, a_mode)
        obs_f, reward, termin_signal, _=self.agent.env.step(action)

        V=self.agent.action_values[obs, action]
        action_f=self.agent.get_action(obs, 'policy')
        V_f=self.agent.action_values[obs_f, action_f]
        priority=torch.abs(reward+self.agent.gamma*V_f-V) #absolute value of TD error as priority
        ep_step=Ep_Step(obs, action, reward, obs_f, priority, termin_signal)
        
        self.replay_buffer.add_item(ep_step)

        #time to update
        if step%update_every==0:
          #update tree of replay buffer
          self.replay_buffer.update(alpha)
          for u_step in range(update_every):
            #get batch data every update step
            batch_data, isrs=self.replay_buffer.sample(batch_size, alpha, beta)
            for idx, data_idx in enumerate(batch_data):
              data=self.replay_buffer.data[data_idx]
              isr=isrs[idx]
              obs=data.obs
              action=data.action
              reward=data.reward
              obs_f=data.obs_f
              ts=data.termin_signal

              #update action values
              V=self.agent.action_values[obs, action]
              action_f=self.agent.get_action(obs_f, 'policy')
              V_f=self.agent.action_values[obs_f, action_f]
              td_error=reward+self.agent.gamma*V_f-V
              
              #update corresponding TD error
              self.agent.action_values[obs, action]=self.agent.action_values[obs, action]+lr*isr*td_error
              #change the td error in leaf nodes
              self.replay_buffer.data[data_idx]=Ep_Step(obs, action, reward, obs_f, torch.abs(td_error), ts)
              
        step+=1
        if termin_signal:
          obs=self.agent.env.reset()
        else:
          obs=obs_f
        if step%steps_per_epoch==1:
          break
      #check performance at end of each epoch
      avg_return=self.check_performance()
      print("Epoch: {:d}, Avg_Return: {:.2f}".format(epoch, avg_return))
      #print(self.agent.action_values)
    return

Ep_Step=collections.namedtuple("Ep_Step", field_names=['obs', 'action', 'reward', 'obs_f', 'priority', 'termin_signal'])

env=blind_cliffwalk=Observation_Wrapper(Blind_Cliffwalk(N=10))
test_env=blind_cliffwalk=Observation_Wrapper(Blind_Cliffwalk(N=10))
agent=Tabular_Agent(env, test_env, gamma=.99, lambd=.97)
tql=Tabular_Q_Learning(agent=agent, per='proportional')
tql.train(batch_size=64, n_epochs=100, steps_per_epoch=40000, start_after=100000, update_after=5000, update_every=100, lr=1e-3, alpha=1, beta=1)

