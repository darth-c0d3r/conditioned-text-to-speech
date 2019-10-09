import torch
from model import Wavenet

X = torch.randn((1,1,1000))
network = Wavenet()

print(X.shape)
print(network(X).shape)