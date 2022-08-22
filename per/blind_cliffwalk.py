import gym
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#blind cliffwalk: on gym environment
class Blind_Cliffwalk(gym.Env):
  def __init__(self, N):
    super(Blind_Cliffwalk, self).__init__()
    self.N=N
    self.observation_space=gym.spaces.Discrete(self.N)
    self.action_space=gym.spaces.Discrete(2)

    self._start=0
    self._horizon=N
    self._step=1
    self._agent_location=0
  
  def _get_obs(self):
    return self._agent_location
  
  def step(self, a_idx):
    if a_idx==0:
      #move to start state w/ zero reward: "wrong move"
      self._agent_location=self._start
      reward=0
    else:
      #move to next state: "right move"
      #if done on last state
      if self._agent_location==(self.N-1):
        reward=1
        self._agent_location=self._start
      else:
        self._agent_location=self._agent_location+1
        reward=0
    obs_f=self._get_obs()

    if self._step==self._horizon or self._agent_location==(self.N-1):
      termin_signal=True
    else:
      self._step+=1
      termin_signal=False
    info=None
    return obs_f, reward, termin_signal, info
  
  def reset(self):
    self._agent_location=self._start
    obs=self._get_obs()
    self._step=1
    return obs

class Observation_Wrapper(gym.ObservationWrapper):
  def __init__(self, env):
    super(Observation_Wrapper, self).__init__(env)

  def observation(self, observation):
    obs=torch.LongTensor([observation]).to(device)
    return obs