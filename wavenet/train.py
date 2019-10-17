import torch
from torch import nn, optim
from torch.autograd import Variable

from util import *
from model import Wavenet
from data import getAudioDataset

def train(model, dataset, loss_fxn, opt, hyperparams, device):

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

			# add loss to the total loss
			total_loss += len(data)*loss.item()

		# print the loss for epoch i if needed
		if epoch % hp.report == 0:
			print("Epoch %d : Loss = %06f" % (epoch, total_loss / float(len(trainloader))))

	# return the trained model
	return model

if __name__ == '__main__':

	# get the required dataset
	folder = "../audio/"
	dataset = getAudioDataset(folder)

	# get the device used
	device = get_device()

	# define the model
	model = Wavenet()
	model = model.to(device)

	# define hyper-parameters
	hp = Hyperparameters()
	hp.lr = 1e-2
	hp.epochs = 10000
	hp.batch_size = 1
	hp.report = 10

	# define loss function and optimizer
	optimizer = optim.Adam(model.parameters(), lr=hp.lr)
	loss_fxn = nn.CrossEntropyLoss()

	# call the train function
	model = train(model, dataset["data"], loss_fxn, optimizer, hp, device)

	# set the sampe_rate and save the model
	model.sample_rate = dataset["rate"]
	save_model(model)

	audio_sample = sample(model, dataset["data"][0][0].shape[1], device)
	save_audio(audio_sample, model.sample_rate)