from util import *
import scipy.io.wavfile as wavfile
from torch.utils.data import Dataset
import os

class AudioDataset(Dataset):

	def __init__(self, root_dir, quantiles=256, non_linear=True):
		
		if root_dir.endswith("/"):
			root_dir = root_dir[:-1]

		self.root_dir = root_dir
		self.files = os.listdir(root_dir)

		# filter the list for .wav files
		self.files = list(filter(lambda name: name.endswith(".wav"), self.files))

		self.quantiles = quantiles
		self.non_linear = non_linear

		self.rate, _ = wavfile.read(self.root_dir+"/"+self.files[0])
		self.warn = False # warns if rates are different

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):

		filename = self.root_dir + "/" + self.files[idx]
		rate, data = wavfile.read(filename)

		if rate != self.rate:
			self.warn = True

		_, indices = quantize_waveform(data, self.quantiles, self.non_linear)
		data = torch.tensor(index2oneHot(indices, self.quantiles))
		indices = torch.tensor(indices).view(-1)

		# data is simply the one hot version of indices
		return (data.float(), indices.long())

def getAudioDataset(root_dir, quantiles=256, non_linear=True):

	dataset = AudioDataset(root_dir, quantiles, non_linear)

	if dataset.warn is True:
		print("Warning: Variable Sample Rates in Dataset")

	return {"data":dataset, "rate":dataset.rate}
