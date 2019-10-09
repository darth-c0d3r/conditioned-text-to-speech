import torch
import torch.nn as nn
from util import *

class Wavenet(nn.Module):
	"""
	Class for a generic Wavenet Model.

	All the layers in the network are Residual Layers.
	Dilation will increase exponentially by a factor of 2.

	"""

	def __init__(self):

		super(Wavenet, self).__init__()

		# --------------------------------#
		# network parameters
		
		num_layers = 32 # number of residual layers in the network
		max_dilation = 8 # after how many layers to reset dilation
		kernel_size = 3
		output_size = 256

		# --------------------------------#

		# initialize the empty list
		self.res_layers = []
		self.kernel_size = kernel_size # required for sampling

		for layer in range(num_layers):

			# calculate dilation and padding
			dilation = 2**(layer%max_dilation)
			padding = kernel_size * dilation - (dilation - 1)

			# 2 conv layers for gated activation unit
			conv1 = nn.Conv1d(1,1,kernel_size,dilation=dilation,padding=padding)
			conv2 = nn.Conv1d(1,1,kernel_size,dilation=dilation,padding=padding)

			# 1x1 conv layer
			conv3 = nn.Conv1d(1,1,1)

			self.res_layers.append((conv1, conv2, conv3))
			
		# output layer will have a categorical loss
		self.output_layer = nn.Conv1d(1,output_size,1)

	def forward(self, X):

		# X.shape = [1,1,n]
		# try [1,k,n] later

		inp_len = X.shape[2]

		for conv1, conv2, conv3 in self.res_layers:

			# gated activation unit + residual connection
			X = X + conv3((torch.tanh(conv1(X))*torch.sigmoid(conv2(X)))[:,:,:inp_len])

		X = torch.softmax(self.output_layer(X), 2)
		return X

	def sample(self, output_len):
		"""
		Used to sample from the learned distribution.
		"""

		waveform = [0.]
		for _ in range(output_len):
			data = torch.tensor(waveform).view(1,1,-1)
			probs = self.forward(data)[0,:,-1]
			idx = torch.multinomial(probs, 1)
			waveform.append(index2normalize(idx))

		return np.array(waveform[1:])


