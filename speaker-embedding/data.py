from util import *
from torch.utils.data import Dataset
import os

class DummyDataset(Dataset):

	def __init__(self, quantiles, num_speakers, samples_per_speaker, sample_length):

		self.quantiles = quantiles # = q
		self.num_speakers = num_speakers # = n
		self.samples_per_speaker = samples_per_speaker # = m
		self.sample_length = sample_length # = l

		self.bits = 16

	def __len__(self):
		# number of possible pairs
		# = (n*m)*(n*m)

		return (self.num_speakers * self.samples_per_speaker) ** 2

	def __getitem__(self, idx):
		
		spkr1 = (idx//(self.num_speakers * self.samples_per_speaker))//self.samples_per_speaker
		spkr2 = (idx %(self.num_speakers * self.samples_per_speaker))//self.samples_per_speaker

		audio1 = 0.001*np.random.randn((self.sample_length)) + (float(2*spkr1)/float(self.num_speakers-1)) - 1.0
		audio2 = 0.001*np.random.randn((self.sample_length)) + (float(2*spkr2)/float(self.num_speakers-1)) - 1.0

		audio1 = np.clip(audio1, -1.0, 0.9999)
		audio2 = np.clip(audio2, -1.0, 0.9999)

		wvfrm1, audio1 = quantize_waveform(audio1 * (2**(self.bits-1)), self.quantiles, non_linear=False)
		wvfrm2, audio2 = quantize_waveform(audio2 * (2**(self.bits-1)), self.quantiles, non_linear=False)

		# visualize_waveform(waveform=wvfrm1)
		# visualize_waveform(waveform=wvfrm2)

		audio1 = index2oneHot(audio1, self.quantiles)
		audio2 = index2oneHot(audio2, self.quantiles)

		return torch.tensor(audio1).t().float(), torch.tensor(audio2).t().float(), torch.tensor(int(spkr1 == spkr2))

def getDummyDataset():

	quantiles = 256
	num_speakers = 4
	samples_per_speaker = 2
	sample_length = 10

	return {"data" : DummyDataset(quantiles, num_speakers, samples_per_speaker, sample_length)}
