import torch
import torchvision.transforms as transforms
import gym
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Obs_Wrapper(gym.ObservationWrapper):
  def __init__(self, env):
    super(Obs_Wrapper, self).__init__(env)
  
  def observation(self, observation):
    #convert to tensor
    observation=torch.FloatTensor(observation).permute(2,0,1).to(device) #3x210x160 image
    #grayscale
    to_grayscale=transforms.Grayscale()
    #110x84 downsampling
    resize=transforms.Resize(size=[110,84])
    obs=resize(observation)
    obs=resize(to_grayscale(observation))
    #84x84 crop -> remove unnecessary top part of the image
    cropped_obs=obs[:,26:,:]
    #normalize into values between 0 and 1
    cropped_obs=cropped_obs/255
    return cropped_obs

class Rew_Wrapper(gym.RewardWrapper):
  def __init__(self, env):
    super(Rew_Wrapper, self).__init__(env)
  
  def reward(self, reward):
    if reward>0:
      return 1.
    elif reward==0:
      return 0.
    else:
      return -1.

def polyak_averaging(module, target_module, polyak):
    #applicable when all layers in module are declared as self.linear_layers=...
    #new_param=target_module*polyak+module*(1-polyak)
    #to match parameters: set polyak to be 0
    new_weights=[]
    new_biases=[]
    for idx, element in enumerate(module.children()):
      if isinstance(element, nn.Linear) or isinstance(element, nn.Conv2d):
        new_weights.append(element.weight*(1-polyak))
        new_biases.append(element.bias*(1-polyak))
    target_idx=0
    for target_element in target_module.children():
      if isinstance(target_element, nn.Linear) or isinstance(target_element, nn.Conv2d):
        new_weights[target_idx]=new_weights[target_idx]+polyak*target_element.weight
        new_biases[target_idx]=new_biases[target_idx]+polyak*target_element.bias
        target_idx+=1
    #inherit new parameters
    inherit_idx=0
    for target_element in target_module.children():
      if isinstance(target_element, nn.Linear) or isinstance(target_element, nn.Conv2d):
        target_element.weight=nn.Parameter(new_weights[inherit_idx])
        target_element.bias=nn.Parameter(new_biases[inherit_idx])
        inherit_idx+=1
    return