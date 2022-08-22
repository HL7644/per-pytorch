import numpy as np
import collections
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Tree_Node():
  def __init__(self, data, parent=None, left_child=None, right_child=None):
    self.data=data
    self.left_child=left_child
    self.right_child=right_child
    self.parent=parent
    
    self.leaf_index=None
  
  def is_leaf(self):
    if self.left_child==None and self.right_child==None:
      return True
    else:
      return False
    
  def __repr__(self):
    return "Data: "+str(self.data)

class Ranking_Prioritized_Replay():
  def __init__(self):
    self.max_length=1000000
    self.n_data=0

    self.data=[] #storing ep_steps

    self.inv_rank_sum=0

    self.min_prior_data=None
    self.max_prior_data=None
  
  def add_item(self, ep_step): #element as a leaf node, item: episode step
    if self.n_data==self.max_length:
      #remove first data
      self.data.pop(0)
      self.n_data-=1
    self.data.append(ep_step)
    self.n_data+=1
    #check max-min data
    if self.max_prior_data==None and self.min_prior_data==None:
      #for the first input step: initialize
      self.max_prior_data=ep_step
      self.min_prior_data=ep_step
    else:
      if ep_step.priority<self.min_prior_data.priority:
        self.min_prior_data=ep_step
      elif ep_step.priority>self.max_prior_data.priority:
        self.max_prior_data=ep_step
    return
  
  def update_inv_rank_sum(self, alpha):
    inv_rank_sum=0
    for rank in range(1, self.n_data+1):
      inv_rank_sum+=(1/rank)**alpha
    self.inv_rank_sum=inv_rank_sum
    return
  
  def get_sampling_intervals(self, batch_size, alpha):
    #approximating cumulative PDF as piecewise linear functions
    intervals=[]
    x_old=1
    if alpha==1:
      for b_idx in range(1, batch_size):
        x=int(np.exp(np.log(self.n_data)*(b_idx/batch_size)))
        intervals.append([x_old,x])
        x_old=x+1
      intervals.append([x_old, self.n_data])
    else:
      for b_idx in range(1, batch_size):
        x=int(((b_idx/batch_size)*(self.n_data**(1-alpha)-1)+1)**(1/(1-alpha)))
        intervals.append([x_old, x])
        x_old=x+1
      intervals.append([x_old, self.n_data])
    #interval values correspond to ranks
    return intervals
  
  def update(self, alpha):
    priorities=torch.zeros(self.n_data).to(device) #tensor used for sorting
    #exponentiate first
    for idx, ep_step in enumerate(self.data):
      priorities[idx]=ep_step.priority
    sorted_idx=torch.argsort(priorities, descending=True)
    new_data=[]
    for rank, data_idx in enumerate(sorted_idx):
      new_data.append(self.data[rank])
    self.data=new_data
    self.update_inv_rank_sum(alpha)
    return

  def sample(self, batch_size, alpha, beta):
    #assume updated data
    #return batch data and corresponding importance sampling ratios
    batch_data=[]
    isrs=[]
    #use piecewise-linear cumulative PDF -> each segment is equiprobable
    interval_size=self.n_data/batch_size
    #get max isr value
    max_isr_prob=((1/self.n_data)**alpha)/self.inv_rank_sum #priorities are 1/ranks -> max_isr_prob: min-priority
    max_isr=(1/(self.n_data*max_isr_prob))**beta
    intervals=self.get_sampling_intervals(batch_size, alpha)
    for interval in intervals:
      #sample uniformly within intervals
      ranking=np.random.randint(low=interval[0], high=interval[1]+1)
      sample_idx=ranking-1 #idx starts from 0 and ranking starts from 1
      batch_data.append(sample_idx)
      priority=(1/ranking)
      prob=priority**alpha/self.inv_rank_sum
      isr=(1/(self.n_data*prob))**beta
      isrs.append(isr)
    return batch_data, isrs