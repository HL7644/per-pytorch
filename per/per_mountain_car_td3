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

from proportional import *
from ranking import *
from atari_setup import polyak_averaging

class Obs_Wrapper(gym.ObservationWrapper):
  def __init__(self, env):
    super(Obs_Wrapper, self).__init__(env)
  
  def observation(self, obs):
    obs=torch.FloatTensor(obs).to(device)
    return obs

class Cont_Deterministic_Policy_Module(nn.Module): #for continuous
  def __init__(self, i_size, hidden_sizes, o_size):
    super(Cont_Deterministic_Policy_Module, self).__init__()
    #create N layers of mlp for output
    layers=[]
    len_h=len(hidden_sizes)
    relu=nn.LeakyReLU()

    first=nn.Linear(i_size, hidden_sizes[0]).to(device)
    nn.init.uniform_(first.weight, -1/np.sqrt(i_size), 1/np.sqrt(i_size))
    nn.init.zeros_(first.bias)
    layers.append(first)
    layers.append(relu)
    for h_idx in range(len_h-1):
      layer=nn.Linear(hidden_sizes[h_idx], hidden_sizes[h_idx+1]).to(device)
      nn.init.uniform_(layer.weight, -1/np.sqrt(hidden_sizes[h_idx]), 1/np.sqrt(hidden_sizes[h_idx]))
      nn.init.zeros_(layer.bias)
      layers.append(layer)
      layers.append(relu)
    last=nn.Linear(hidden_sizes[-1], o_size).to(device)
    nn.init.uniform_(last.weight, -(3e-3), 3e-3)
    nn.init.uniform_(last.bias, -(3e-3), 3e-3)
    layers.append(last)
    layers.append(nn.Tanh())
    #last activ ftn: tanh to limit values
    self.linear_layers=nn.Sequential(*list(layers))
  
  def forward(self, observation):
    action=self.linear_layers(observation)
    return action

class TD3_Value_Module(nn.Module):
  def __init__(self, i_size, a_dim, hidden_sizes):
    super(TD3_Value_Module, self).__init__()
    #create N layers of mlp for output, o_size=1
    layers=[]
    len_h=len(hidden_sizes)
    relu=nn.ReLU()
    first=nn.Linear(i_size+a_dim, hidden_sizes[0]).to(device)
    f=i_size+a_dim
    nn.init.uniform_(first.weight, -1/np.sqrt(f), 1/np.sqrt(f))
    nn.init.zeros_(first.bias)
    layers.append(first)
    layers.append(relu)
    for h_idx in range(len_h-1):
      linear=nn.Linear(hidden_sizes[h_idx], hidden_sizes[h_idx+1]).to(device)
      f=hidden_sizes[h_idx]
      nn.init.uniform_(linear.weight, -1/np.sqrt(f), 1/np.sqrt(f))
      nn.init.zeros_(linear.bias)
      layers.append(linear)
      layers.append(relu)
    last=nn.Linear(hidden_sizes[-1], 1).to(device)
    nn.init.uniform_(first.weight, -(3e-3), 3e-3)
    nn.init.uniform_(first.bias, -(3e-3), 3e-3)
    layers.append(last)
    #last layer activ: Identity
    self.linear_layers=nn.Sequential(*list(layers))
  
  def forward(self, observation, action):
    #add action
    fv=torch.cat((observation, action), dim=0).to(device)
    #forward pass through rest of the layers
    value=self.linear_layers(fv)
    return value

Ep_Step=collections.namedtuple("Ep_Step", field_names=['obs', 'action', 'reward', 'obs_f', 'priority', 'termin_signal'])

class TD3_Agent():
  def __init__(self, env, test_env):
    self.env=env
    self.test_env=test_env
    self.a_low=torch.FloatTensor(self.env.action_space.low).to(device)
    self.a_high=torch.FloatTensor(self.env.action_space.high).to(device)

    self.i_size=self.env.observation_space.shape[0]
    self.a_dim=self.env.action_space.shape[0]
    #policy module(actor)
    self.pm=Cont_Deterministic_Policy_Module(self.i_size, [400,300], self.a_dim)
    self.target_pm=Cont_Deterministic_Policy_Module(self.i_size, [400,300], self.a_dim)
    #use same paramters
    polyak_averaging(self.pm, self.target_pm, polyak=0)
    #value modules(critic)
    self.vm1=TD3_Value_Module(self.i_size, self.a_dim, [400,300])
    self.target_vm1=TD3_Value_Module(self.i_size, self.a_dim, [400,300])
    polyak_averaging(self.vm1, self.target_vm1, polyak=0)

    self.vm2=TD3_Value_Module(self.i_size, self.a_dim, [400,300])
    self.target_vm2=TD3_Value_Module(self.i_size, self.a_dim, [400,300])
    polyak_averaging(self.vm2, self.target_vm2, polyak=0)
  
  def add_noise(self, action, std): 
    noise=torch.normal(torch.zeros(self.a_dim), torch.full([self.a_dim], std)).to(device)
    action=action+noise
    return action

  #used for training: only uses current modules
  def get_action(self, obs, a_mode, std):
    if a_mode=='random':
      action=torch.FloatTensor(self.env.action_space.sample()).to(device)
    elif a_mode=='policy':
      act=self.pm(obs)
      action=self.add_noise(act, std)
    else:
      #action w/o noise for test
      assert a_mode=='test', 'invalid action mode'
      action=self.pm(obs)

    #clip action within action boundaries
    action=torch.clamp(action, self.a_low, self.a_high)
    return action

class TD3():
  def __init__(self, agent, per):
    self.agent=agent
    if per=='proportional':
      self.replay_buffer=Proportional_Prioritized_Replay()
    else:
      assert per=='ranking', 'invalid replay type'
      self.replay_buffer=Ranking_Prioritized_Replay()
  
  def check_performance(self):
    a_mode='test'
    #w.r.t test env.: run 10 episodes
    avg_return=0
    avg_len_ep=0
    for idx in range(1,10+1):
      obs=self.agent.test_env.reset()
      len_ep=1
      acc_rew=0
      ep_data=[]
      while True:
        action=self.agent.get_action(obs, a_mode, std=0).detach().cpu().numpy()
        obs_f, reward, termin_signal, _=self.agent.test_env.step(action)
        acc_rew+=reward
        if termin_signal:
          break
        else:
          len_ep+=1
          obs=obs_f
      #incremental update
      avg_return=avg_return+(acc_rew-avg_return)/idx
      avg_len_ep=avg_len_ep+(len_ep-avg_len_ep)/idx
    return avg_return, avg_len_ep
  
  
  def get_policy_loss(self, batch_data):
    #update in a direction of maximizing vm1 estimates
    batch_size=len(batch_data)
    policy_loss=torch.FloatTensor([0]).to(device)

    for sample_idx in batch_data:
      ep_step=self.replay_buffer.data[sample_idx]
      obs=ep_step.obs
      #use action w/o noise for policy update
      action=self.agent.pm(obs)
      Q=self.agent.vm1(obs, action)
      policy_loss=policy_loss-Q #negative for SGA
    policy_loss=policy_loss/batch_size
    return policy_loss
  
  def get_value_loss(self, gamma, batch_data, isrs, target_action_std, target_noise_thresh, vm_idx):
    if vm_idx==1:
      vm=self.agent.vm1
    else:
      assert vm_idx==2, 'Invalid value module index'
      vm=self.agent.vm2
    
    batch_size=len(batch_data)
    value_loss=torch.FloatTensor([0]).to(device)
    for idx, sample_idx in enumerate(batch_data):
      isr=isrs[idx]
      ep_step=self.replay_buffer.data[sample_idx]

      obs=ep_step.obs
      action=ep_step.action.detach()
      reward=ep_step.reward
      obs_f=ep_step.obs_f
      termin_signal=ep_step.termin_signal

      #get target action
      ta=self.agent.target_pm(obs_f)
      #generalize target action by adding clipped noise
      target_noise=torch.normal(torch.zeros(self.agent.a_dim), torch.full([self.agent.a_dim], target_action_std)).to(device)
      clipped_noise=torch.clamp(target_noise, -target_noise_thresh, target_noise_thresh)
      ta=ta+clipped_noise
      #clip target_action+noise into action boundaries
      target_action=torch.clamp(ta, self.agent.a_low, self.agent.a_high)

      #create target
      q1_f=self.agent.target_vm1(obs_f, target_action)
      q2_f=self.agent.target_vm2(obs_f, target_action)
      min_q_f=min(q1_f,q2_f)
      #target doesn't require gradient
      target=(reward+gamma*(1-termin_signal)*min_q_f).detach()
      Q=vm(obs, action)
      #multiply isr to adjust step-size for value update
      value_loss=value_loss+isr*(target-Q)**2

      #adjust priorities for selected samples
      #priorities: computed w.r.t vm1
      if vm_idx==1:
        action_f=self.agent.get_action(obs_f, 'test', std=0)
        qf=self.agent.vm1(obs_f, action_f)
        priority=torch.abs(reward+qf-Q).detach()
        self.replay_buffer.data[sample_idx]=self.replay_buffer.data[sample_idx]._replace(priority=priority)

    value_loss=value_loss/batch_size
    return value_loss
  
  def train(self, gamma, batch_size, n_epochs, steps_per_epoch, start_after, update_after, update_every, 
            update_actor_every, action_std, target_action_std, target_noise_thresh, p_lr, v_lr, polyak, alpha, beta):
    #optimizers
    policy_optim=optim.Adam(self.agent.pm.parameters(), lr=p_lr)
    value_optim1=optim.Adam(self.agent.vm1.parameters(), lr=v_lr)
    value_optim2=optim.Adam(self.agent.vm2.parameters(), lr=v_lr)

    #train process
    obs=self.agent.env.reset()
    step=1
    a_mode='random'
    update=False
    for epoch in range(1, n_epochs+1):
      while True:
        if step>start_after:
          a_mode='policy'
        if step>update_after:
          update=True
        
        #action doesn't require grad for progression
        action=self.agent.get_action(obs, a_mode, action_std)
        obs_f, reward, termin_signal, _=self.agent.env.step(action.detach().cpu().numpy())
        if termin_signal:
          if obs_f[0]>0.45:
            #real termination of reaching goal
            termin_signal=1
          else:
            #just reaching horizon
            termin_signal=0
          obs_f=self.agent.env.reset()
        
        #compute priority w.r.t vm1
        action_f=self.agent.get_action(obs_f, 'policy', action_std)
        Q=self.agent.vm1(obs, action)
        Q_f=self.agent.vm1(obs_f, action_f)
        tde=reward+gamma*Q_f-Q
        priority=torch.abs(tde).detach() #remove gradient in priority to avoid graphing twice
        ep_step=Ep_Step(obs, action, reward, obs_f, priority, termin_signal)
        self.replay_buffer.add_item(ep_step)

        if step%update_every==0 and update:
          self.replay_buffer.update(alpha)
          for u_step in range(1, update_every+1):
            #"update actor every" -> rate of actor update
            batch_data, isrs=self.replay_buffer.sample(batch_size, alpha, beta)
            #update both critics w.r.t same target
            value_loss1=self.get_value_loss(gamma, batch_data, isrs, target_action_std, target_noise_thresh, vm_idx=1)

            value_optim1.zero_grad()
            value_loss1.backward()
            value_optim1.step()

            value_loss2=self.get_value_loss(gamma, batch_data, isrs, target_action_std, target_noise_thresh, vm_idx=2)

            value_optim2.zero_grad()
            value_loss2.backward()
            value_optim2.step()

            if u_step%update_actor_every==0:
              #update actor
              policy_loss=self.get_policy_loss(batch_data)
              
              policy_optim.zero_grad()
              policy_loss.backward()
              policy_optim.step()

              #update target networks when updating the policy
              polyak_averaging(self.agent.pm, self.agent.target_pm, polyak)
              polyak_averaging(self.agent.vm1, self.agent.target_vm1, polyak)
              polyak_averaging(self.agent.vm2, self.agent.target_vm2, polyak)
        obs=obs_f
        step=step+1
        if step%steps_per_epoch==1:
          break
      #per epoch performance measure
      avg_acc_rew, avg_len_ep=self.check_performance()
      print("Epoch: {:d}, Avg_Return, {:.3f}, Avg_Ep_Length: {:.2f}".format(epoch, avg_acc_rew, avg_len_ep))
    return

cont_mc=Obs_Wrapper(gym.make('MountainCarContinuous-v0'))
cont_mc_t=Obs_Wrapper(gym.make('MountainCarContinuous-v0'))
agent=TD3_Agent(cont_mc, cont_mc_t)
td3=TD3(agent, 'proportional')

td3.train(gamma=0.99, batch_size=64, n_epochs=100, steps_per_epoch=4000, start_after=10000, update_after=5000, update_every=50, 
          update_actor_every=2, action_std=0.1, target_action_std=0.2, target_noise_thresh=0.5, p_lr=1e-3, v_lr=1e-3, polyak=0.995, alpha=1, beta=1)

