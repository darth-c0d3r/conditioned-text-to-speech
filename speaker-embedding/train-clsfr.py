import os

import torch
from torch import nn, optim
from torch.autograd import Variable

from util import *
from model import SpeakerEmbedding
from model import Classifier
from data import getSpeakerDataset

def train(embd, clsr, dataset, loss_fxn, opt, scd, hyperparams, device, plot):
	"""
	Parameters:
	embd: The model that gives the speaker embeddings
	clsr: The model that classifies the speaker
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
	clsr.train()


	# iterate epochs number of times
	for epoch in range(1, hyperparams.epochs+1):

		# put the data into a dataloader
		trainloader = torch.utils.data.DataLoader(dataset, batch_size=hyperparams.batch_size, shuffle=True)

		total_loss = 0.
		total_correct = 0

		# iterate over all batches
		for spkr, target in trainloader:

			spkr, target = Variable(spkr.to(device)), Variable(target.to(device))

			# zero out the gradients
			embd.zero_grad()
			clsr.zero_grad()
			opt.zero_grad()

			# get the output and loss
			e = embd(spkr)
			out = clsr(e)
			pred = torch.argmax(out,1).view(target.shape)
			loss = loss_fxn(out, target)

			# get the gradients and update params
			loss.backward()
			opt.step()
			scd.step()

			# add loss to the total loss
			total_loss += spkr.shape[0]*loss.item()
			total_correct += torch.sum(pred == target)

		# print the loss for epoch i if needed
		if epoch % hyperparams.report == 0:
			print("Epoch %d : Loss = %08f | Accuracy = %d/%d (%06f)" % (epoch, total_loss / float(len(trainloader)), \
				total_correct, len(trainloader), float(total_correct)/float(len(trainloader))))


			# make the plot if needed
			if plot is True:
				plotter.plot('loss', 'train', 'Training Loss', epoch, total_loss / float(len(trainloader)))

		if epoch % 1 == 0:
			save_model(embd, "embd_clsr_%d.pt"%(epoch))

	# return the trained model
	return embd, clsr

def main():

	# get the required dataset
	folder = "../libri-speech"
	dataset = getSpeakerDataset(folder)

	# get the device used
	device = get_device(False)

	# define the models
	embd = SpeakerEmbedding(dataset["data"].num_features)
	clsr = Classifier(dataset["data"].num_speakers)
	embd = embd.to(device)
	clsr = clsr.to(device)

	# define hyper-parameters
	hp = Hyperparameters()
	hp.lr = 1e-4
	hp.epochs = 200
	hp.batch_size = 1
	hp.report = 1

	# define optimizer, scheduler, and loss function
	optimizer = optim.Adam(list(embd.parameters())+list(clsr.parameters()), lr=hp.lr)
	scheduler = optim.lr_scheduler.StepLR(optimizer, hp.epochs//2, gamma=0.1)
	loss_fxn = nn.CrossEntropyLoss()

	# call the train function
	plot = False # Use Visdom to plot the Training Loss Curve
	embd, _ = train(embd, clsr, dataset["data"], loss_fxn, optimizer, scheduler, hp, device, plot)
	print("Training over.")
	# save the embedding model
	save_model(embd, "embd_clsr2.pt")

if __name__=='__main__':
	main()