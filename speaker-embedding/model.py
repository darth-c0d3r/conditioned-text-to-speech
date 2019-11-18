import torch
import torch.nn as nn

class SpeakerEmbedding(nn.Module):
	"""
	Class for an LSTM based Speaker Embedding Network.

	"""

	def __init__(self, input_size):

		super(SpeakerEmbedding, self).__init__()

		# --------------------------------#
		# network parameters
		self.input_size = input_size # equal to the number of quantiles
		self.num_layers = 3 # number of stacked LSTM layers
		self.hidden_size = 128 # size of hidden state
		self.embedding_size = 128 # size of speaker representation
		# --------------------------------#

		self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
		self.output_layer = nn.Linear(self.hidden_size, self.embedding_size)

	def forward(self, X):
		# input : batch_size, seq_size, feat_size

		batch_size = X.shape[0]

		# initialize the hidden state and cell state
		# self.num_layers * 1, because num_directions = 1
		h0 = torch.randn((self.num_layers, batch_size, self.hidden_size))
		c0 = torch.randn((self.num_layers, batch_size, self.hidden_size))

		_, (hn, _) = self.lstm(X, (h0, c0))
		output = self.output_layer(hn[-1])
		return output

class Discriminator(nn.Module):
	"""
	Class for a simple discriminator.
	Takes in 2 inputs and checks if they're from same domain.

	"""

	def __init__(self):

		super(Discriminator, self).__init__()

		# --------------------------------#
		# network parameters
		self.embedding_size = 128 # embedding size
		fc = [64] # dimensions of fc hidden layers
		# --------------------------------#
		
		fc = [self.embedding_size * 2] + fc

		# initialize the empty list of module list
		self.fc_layers = nn.ModuleList()

		for i in range(len(fc)-1):
			self.fc_layers.append(nn.Linear(fc[i], fc[i-1]))

		# output layer
		self.output_layer = nn.Linear(fc[-1], 2)

	def forward(self, Va, Vb):
		# Va and Vb are the two speaker embeddings
		# Va.shape = [batch-size, self.embedding_size]
			
		# concatenate the two embeddings
		V = torch.cat([Va, Vb], 1)

		# iterate over all fully connected layers
		for fc_layer in self.fc_layers:
			V = torch.relu(fc_layer(V))

		V = self.output_layer(V)

		return V

