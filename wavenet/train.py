import os
import scipy.io.wavfile as wavfile

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable

from util import *
from model import Wavenet
from model import Convnet

folder = "../audio/"
files = os.listdir(folder)

device = get_device()

rate, data = wavfile.read(folder+files[0])
data = np.zeros(data.shape)
data = data[:10]

data, indices = quantize_waveform(data)


print("Using file: %s"%(files[0]))

epochs = 1000

network = Convnet()
network = network.to(device)

optimizer = optim.Adam(network.parameters(), lr=1e-4)

data = torch.tensor(data).float().view(1,1,-1)
target = torch.tensor(indices).view(1,-1)

data = torch.zeros(data.shape)-1
target = torch.zeros(target.shape).long()

for epoch in range(epochs):

	data, target = Variable(data).to(device), Variable(target).to(device)

	out = network(data)
	loss = F.cross_entropy(out, target)

	if(epoch%50 == 0):
		print(float(loss))

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

print("Generating sample.")

network.eval()
sample = sample(network, data.shape[2], device)
wavfile.write(folder+"sample_"+files[0],rate,sample)
print(sample)