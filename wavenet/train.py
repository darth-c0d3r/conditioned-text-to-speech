import os

import torch
from torch import nn, optim
from torch.autograd import Variable

from util import *
from sample import sample
from model import Wavenet
from data import getAudioDataset

def train(model, dataset, loss_fxn, opt, scd, hyperparams, device, plot):
	"""
	Parameters:
	model: The network which is going to be trained
	dataset: The dataset on which the model is to be trained
	loss_fxn: The Loss Function used
	opt: Optimzer
	scd: The Learning Rate Scheduler
	hyperparams: Object of class Hyperparameter containing hyperparams
	device: The device being used to train (CPU/GPU)
	plot: bool value to indicate if plotting is to be done
	"""

	# set up the plotting script
	if plot is True:
		os.system("python3 -m visdom.server")
		plotter = VisdomLinePlotter("Wavenet")

	model.train()

	# iterate epochs number of times
	for epoch in range(1, hyperparams.epochs+1):

		# put the data into a dataloader
		trainloader = torch.utils.data.DataLoader(dataset, batch_size=hyperparams.batch_size, shuffle=True)

		total_loss = 0.

		# iterate over all batches
		for data, target in trainloader:

			data, target = Variable(data.to(device)), Variable(target.to(device))

			# zero out the gradients
			model.zero_grad()
			opt.zero_grad()

			# get the output and loss
			out = model(data)
			loss = loss_fxn(out, target)

			# get the gradients and update params
			loss.backward()
			opt.step()
			scd.step()

			# add loss to the total loss
			total_loss += len(data)*loss.item()

		# print the loss for epoch i if needed
		if epoch % hp.report == 0:
			print("Epoch %d : Loss = %08f" % (epoch, total_loss / float(len(trainloader))))

			# make the plot if needed
			if plot is True:
				plotter.plot('loss', 'train', 'Training Loss', epoch, total_loss / float(len(trainloader)))

	# return the trained model
	return model

if __name__ == '__main__':

	# get the required dataset
	folder = "../audio_small/"
	dataset = getAudioDataset(folder)

	# get the device used
	device = get_device()

	# define the model
	model = Wavenet()
	model = model.to(device)
	model.device = device

	# define hyper-parameters
	hp = Hyperparameters()
	hp.lr = 1e-2
	hp.epochs = 2000
	hp.batch_size = 1
	hp.report = 10

	# define optimizer, scheduler, and loss function
	optimizer = optim.Adam(model.parameters(), lr=hp.lr)
	scheduler = optim.lr_scheduler.StepLR(optimizer, hp.epochs//4, gamma=0.25)
	loss_fxn = nn.CrossEntropyLoss()

	# call the train function
	plot = False # Use Visdom to plot the Training Loss Curve
	model = train(model, dataset["data"], loss_fxn, optimizer, scheduler, hp, device, plot)

	# set the sampe_rate and save the model
	model.sample_rate = dataset["rate"]
	save_model(model, "wavenet1.pt")

	# Free Samples (with the training) [Sorry xD]
	audio_sample = sample(model, dataset["data"][0][0].shape[1], device)
	save_audio(audio_sample, model.sample_rate, "sample1.wav")