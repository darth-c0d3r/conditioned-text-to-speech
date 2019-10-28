import os

import torch
from torch import nn, optim
from torch.autograd import Variable

from util import *
from model import SpeakerEmbedding
from model import Discriminator
from data import getSpeakerDataset

def train(embd, disc, dataset, loss_fxn, opt, scd, hyperparams, device, plot):
	"""
	Parameters:
	embd: The model that gives the speaker embeddings
	disc: The model that discriminates between 2 embeddings
	dataset: The dataset on which the model is to be trained
	loss_fxn: The Loss Function used
	opt: Optimzer
	scd: The Learning Rate Scheduler
	hyperparams: Object of class Hyperparameter containing hyperparams
	device: The device being used to train (CPU/GPU)
	plot: bool value to indicate if plotting is to be done
	input should be of shape (seq_len, batch, input_size)
	"""

	# set up the plotting script
	if plot is True:
		os.system("python3 -m visdom.server")
		plotter = VisdomLinePlotter("SpeakerEmbedding")

	embd.train()
	disc.train()

	# iterate epochs number of times
	for epoch in range(1, hyperparams.epochs+1):

		# put the data into a dataloader
		trainloader = torch.utils.data.DataLoader(dataset, batch_size=hyperparams.batch_size, shuffle=True)

		total_loss = 0.

		# iterate over all batches
		for spkr1, spkr2, target in trainloader:

			spkr1, spkr2, target = Variable(spkr1.to(device)), Variable(spkr2.to(device)), Variable(target.to(device))

			# zero out the gradients
			embd.zero_grad()
			disc.zero_grad()
			opt.zero_grad()

			# get the output and loss
			e1 = embd(spkr1)
			e2 = embd(spkr2)
			out = disc(e1, e2)
			loss = loss_fxn(out, target)

			# get the gradients and update params
			loss.backward()
			opt.step()
			scd.step()

			# add loss to the total loss
			total_loss += spkr1.shape[0]*loss.item()

		# print the loss for epoch i if needed
		if epoch % hp.report == 0:
			print("Epoch %d : Loss = %08f" % (epoch, total_loss / float(len(trainloader))))

			# make the plot if needed
			if plot is True:
				plotter.plot('loss', 'train', 'Training Loss', epoch, total_loss / float(len(trainloader)))

	# return the trained model
	return embd, disc

if __name__ == '__main__':

	# get the required dataset
	# folder = "../audio/"
	dataset = getSpeakerDataset("../audio/")

	# get the device used
	device = get_device()

	# define the models
	embd = SpeakerEmbedding()
	disc = Discriminator()
	embd = embd.to(device)
	disc = disc.to(device)

	# define hyper-parameters
	hp = Hyperparameters()
	hp.lr = 1e-4
	hp.epochs = 1000
	hp.batch_size = 1
	hp.report = 1

	# define optimizer, scheduler, and loss function
	optimizer = optim.Adam(list(embd.parameters())+list(disc.parameters()), lr=hp.lr)
	scheduler = optim.lr_scheduler.StepLR(optimizer, hp.epochs//4, gamma=1)
	loss_fxn = nn.CrossEntropyLoss()

	# call the train function
	plot = False # Use Visdom to plot the Training Loss Curve
	embd, _ = train(embd, disc, dataset["data"], loss_fxn, optimizer, scheduler, hp, device, plot)

	# save the embedding model
	save_model(embd, "embd1.pt")
