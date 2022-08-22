import torch
import collections
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from proportional import Proportional_Prioritized_Replay
from ranking import Ranking_Prioritized_Replay

Ep_Step=collections.namedtuple("Ep_Step", field_names=['obs', 'action', 'reward', 'obs_f', 'priority', 'termin_signal'])

#Verify Sampling Method of both Variants: use 100 dummy data
#1) Proportional
N=100
alpha=0.4242
prop=Proportional_Prioritized_Replay()
trues=[]
counts=torch.zeros(N).to(device)
#generate dummy data
for value in range(1,N+1):
  es=Ep_Step(obs=None, action=None, reward=None, obs_f=None, priority=torch.FloatTensor([value]).to(device), termin_signal=None)
  prop.add_item(es)
  trues.append(value**alpha)
trues=torch.FloatTensor(trues).to(device)
true_probs=trues/torch.sum(trues)
prop.update(alpha=alpha)
#true data
batch_data, _=prop.sample(10000, alpha=alpha, beta=1)
for idx in batch_data:
  counts[idx]+=1
probs=counts/torch.sum(counts)
print(true_probs)
print(probs)

#2) Ranking
ranking=Ranking_Prioritized_Replay()
N=100
alpha=0.5
trues=[]
counts=torch.zeros(N).to(device)
#generate dummy data
for value in range(N, 0, -1):
  rank=N+1-value
  es=Ep_Step(obs=None, action=None, reward=None, obs_f=None, priority=torch.FloatTensor([value]).to(device), termin_signal=None)
  ranking.add_item(es)
  trues.append((1/rank)**alpha)
trues=torch.FloatTensor(trues).to(device)
true_probs=trues/torch.sum(trues)
ranking.update(alpha=alpha)
#true data
for _ in range(10000):
  batch_data, _=ranking.sample(10, alpha=alpha, beta=1)
  for idx in batch_data:
    counts[idx]+=1
probs=counts/torch.sum(counts)
print(true_probs)
print(probs)