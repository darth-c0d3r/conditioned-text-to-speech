import torch
from model import Wavenet
import os
import scipy.io.wavfile as wavfile
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from util import *

folder = "../audio/"
files = os.listdir(folder)

rate, data = wavfile.read(folder+files[0])
data, indices = quantize_waveform(data)

epochs = 100

network = Wavenet()
optimizer = optim.Adam(network.parameters(), lr=10)

data = torch.tensor(data).float().view(1,1,-1)
target = torch.tensor(indices).view(1,-1)

for epoch in range(epochs):

	data, target = Variable(data), Variable(target)

	out = network(data)
	loss = F.cross_entropy(out, target)

	print(float(loss))

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()