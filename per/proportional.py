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

class Proportional_Prioritized_Replay():
  def __init__(self):
    self.data=[]
    self.max_length=1000000 #max length of buffer
    self.n_data=0 #no data initially

    self.tree_data=collections.deque([]) #sum tree data structure
    self.len_tree=len(self.tree_data)-1 #excluding the None data

    self.min_prior_data=None #used when adding an item
    self.max_prior_data=None #used to compute max isr
  
  #start update w/ only even number of leafs
  def update(self, alpha):
    assert self.n_data%2==0, "# of data is odd"
    #clear existing trees
    self.tree_data=collections.deque([]) 
    #first add leaf-nodes: process the first layer
    old_tree_layer=[]
    new_tree_layer=[]
    for leaf_idx, ep_step in enumerate(self.data):
      #update max, min prior data
      if self.max_prior_data.priority<ep_step.priority:
        self.max_prior_data=ep_step
      elif self.min_prior_data.priority>ep_step.priority:
        self.min_prior_data=ep_step
      
      #node data: episode steps
      step_node=Tree_Node(data=ep_step)
      step_node.leaf_index=leaf_idx
      #exponentiate priority by alpha
      step_node.data=step_node.data._replace(priority=step_node.data.priority**alpha)
      self.tree_data.append(step_node)
      old_tree_layer.append(step_node)
    idx=0
    len_old=len(old_tree_layer)
    while True:
      #priorities here for the new layers are already exponentiated
      new_data=old_tree_layer[idx].data.priority+old_tree_layer[idx+1].data.priority
      new_node=Tree_Node(data=new_data, left_child=old_tree_layer[idx], right_child=old_tree_layer[idx+1])
      old_tree_layer[idx].parent=new_node
      old_tree_layer[idx+1].parent=new_node
      new_tree_layer.append(new_node)
      if idx+2>=len_old:
        break
      else:
        idx=idx+2
    #add new tree layer to tree data
    len_new=len(new_tree_layer)
    for new_idx in range(len_new-1, -1, -1):
      self.tree_data.appendleft(new_tree_layer[new_idx])
    #update old tree layer
    old_tree_layer=new_tree_layer

    #build upon the layers
    while True:
      #build new tree layer
      #node data: sum of tdes
      new_tree_layer=[]
      len_old=len(old_tree_layer)
      idx=0
      while True:
        if idx+1>=len_old:
          #case of ending with a single data
          new_node=old_tree_layer[idx]
        else:
          new_data=old_tree_layer[idx].data+old_tree_layer[idx+1].data
          new_node=Tree_Node(data=new_data, left_child=old_tree_layer[idx], right_child=old_tree_layer[idx+1])
          old_tree_layer[idx].parent=new_node
          old_tree_layer[idx+1].parent=new_node
        new_tree_layer.append(new_node)
        if idx+2>=len_old:
          break
        else:
          idx=idx+2
      #add new tree layer to tree data
      len_new=len(new_tree_layer)
      for new_idx in range(len_new-1, -1, -1):
        self.tree_data.appendleft(new_tree_layer[new_idx])
      #if top node has been reached
      if len_new==1:
        break
      #update old tree layer
      old_tree_layer=new_tree_layer

    self.tree_data.appendleft(None)
    self.len_tree=len(self.tree_data)-1
    return

  def add_item(self, ep_step): #element as a leaf node, item: episode step
    if self.n_data==self.max_length:
      #remove first data
      self.data.pop(0)
      self.n_data-=1
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
    #add ep_step w/ max priority available, assume: priority decreases after each update
    ep_step=ep_step._replace(priority=self.max_prior_data.priority)
    self.data.append(ep_step)
    self.n_data+=1
    return

  def sample(self, batch_size, alpha, beta): #sample from current sum-tree
    #assume the tree is updated
    len_tree=len(self.tree_data)
    batch_data=[] #return batch data of leaf node indices
    isrs=[]
    interval_size=(self.tree_data[1].data/batch_size).item()
    #get max isr value: max isr comes from min-priority step
    max_isr_prob=(self.min_prior_data.priority**alpha)/self.tree_data[1].data
    max_isr=(1/(self.n_data*max_isr_prob))**beta
    for b_idx in range(batch_size):
      sample_number=np.random.uniform(low=b_idx*interval_size, high=(b_idx+1)*interval_size)
      #search based on sample number: starting at idx=1
      node=self.tree_data[1] #total sum
      while True:
        if node.is_leaf():
          batch_data.append(node.leaf_index)
          #compute isr
          prob=node.data.priority**alpha/self.tree_data[1].data
          isr=(1/(self.n_data*prob))**beta
          isr=isr/max_isr
          isrs.append(isr)
          break
        else:
          left_node=node.left_child
          if left_node.is_leaf():
            l_val=left_node.data.priority.item()
          else:
            l_val=left_node.data.item()
          if sample_number<l_val:
            node=node.left_child
          else:
            sample_number=sample_number-l_val
            node=node.right_child
    return batch_data, isrs