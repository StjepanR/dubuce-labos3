from torch.utils.data import Dataset, DataLoader
import torch


class NLPDataset(Dataset):
	def __init__(self):
		super().__init__()
		self.instances = []
		self.text_frequency = {}
		self.label_frequency = {}
		self.text_vocab = None
		self.label_vocab = None

	def from_file(self, path):
		with open(path, "r") as f:
			for line in f:
				instance_text, instance_label = line.strip().split(", ")

				for instance_word in instance_text.split(" "):
					if instance_word not in self.text_frequency.keys():
						self.text_frequency[instance_word] = 1
					else:
						self.text_frequency[instance_word] = self.text_frequency[instance_word] + 1

				if instance_label not in self.label_frequency.keys():
					self.label_frequency[instance_label] = 1
				else:
					self.label_frequency[instance_label] = self.label_frequency[instance_label] + 1

				self.instances.append((instance_text.strip().split(" "), instance_label.strip()))

		self.text_vocab = Vocab(frequencies=self.text_frequency)
		self.label_vocab = Vocab(frequencies=self.label_frequency, add_extra=False)

	def __getitem__(self, idx):
		item = torch.tensor([self.text_vocab.encode(instance_text) for instance_text in self.instances[idx][0]]), self.label_vocab.encode(self.instances[idx][1])
		return item

	def __len__(self):
		return len(self.instances)


class Vocab():
	def __init__(self, frequencies, max_size=-1, min_freq=0, add_extra=True):

		freq = {key: value for key, value in frequencies.items() if value >= min_freq}
		freq = sorted(freq.items(), key=lambda item: item[1], reverse=True)

		if add_extra:
			freq.insert(0, ("<PAD>", 0))
			freq.insert(1, ("<UNK>", 0))

		if max_size != -1:
			freq = freq[:max_size]

		self.stoi = {}
		self.itos = {}

		idx = 0
		for key in freq:
			self.stoi[key[0]] = idx
			self.itos[idx] = key[0]
			idx += 1

	def encode(self, token):
		if type(token) is list:
			return torch.tensor([self.encode(item) for item in token])
		else:
			return torch.tensor(self.stoi.get(token, self.stoi.get("<UNK>", 0)))

	def get_embedding_matrix(self):
		d = 300

		embeddings = torch.normal(mean=0, std=1, size=(len(self.stoi), d))
		embeddings[0] = torch.tensor(0)

		return torch.nn.Embedding.from_pretrained(embeddings, padding_idx=0, freeze=False)

	def get_embedding_matrix_glove(self):
		d = 300

		embeddings = torch.normal(mean=0, std=1, size=(len(self.stoi), d))
		embeddings[0] = torch.tensor(0)

		with open("./data/sst_glove_6b_300d.txt", "r") as f:
			for line in f:
				embedding = line.strip().split(" ")
				word = embedding.pop(0)
				row = self.stoi.get(word, self.stoi.get("<UNK>"))
				embedding = torch.tensor([float(embed) for embed in embedding])
				embeddings[row] = embedding

		return torch.nn.Embedding.from_pretrained(embeddings, padding_idx=0, freeze=False)

def collate_fn(batch):
	"""
	Arguments:
	  Batch:
		list of Instances returned by `Dataset.__getitem__`.
	Returns:
	  A tensor representing the input batch.
	"""

	texts, labels = zip(*batch)  # Assuming the instance is in tuple-like form
	lengths = torch.tensor([len(text) for text in texts])  # Needed for later
	# Process the text instances
	return texts, labels, lengths

def pad_collate_fn(batch, pad_index=0):
	texts, labels, lengths = collate_fn(batch=batch)
	return torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=pad_index), labels, lengths

def test_method1(train_dataset, text_vocab, label_vocab):
	instance_text, instance_label = train_dataset.instances[3]
	print(f"Text: {instance_text}")  # Text: ['yet', 'the', 'act', 'is', 'still', 'charming', 'here']
	print(f"Label: {instance_label}")  # Label: positive

	print(
		f"Numericalized text: {text_vocab.encode(instance_text)}")  # Numericalized text: tensor([189,   2, 674,   7, 129, 348, 143])
	print(f"Numericalized label: {label_vocab.encode(instance_label)}")  # Numericalized label: tensor(0)


def test_method2(frequencies):
	print(frequencies['the'])  # 5954
	print(frequencies['a'])  # 4361
	print(frequencies['and'])  # 3831
	print(frequencies['of'])  # 3631

	text_vocab = Vocab(frequencies, max_size=-1, min_freq=0)
	print(text_vocab.stoi['<PAD>'])  # 0
	print(text_vocab.stoi['<UNK>'])  # 1
	print(text_vocab.stoi['the'])  # 2
	print(text_vocab.stoi['a'])  # 3
	print(text_vocab.stoi['and'])  # 4
	print(text_vocab.stoi['of'])  # 5

	print(text_vocab.itos[0])  # <PAD>
	print(text_vocab.itos[1])  # <UNK>
	print(text_vocab.itos[2])  # the
	print(text_vocab.itos[3])  # a
	print(text_vocab.itos[4])  # and
	print(text_vocab.itos[5])  # of
	print(len(text_vocab.itos))  # 14806

	print(text_vocab.stoi['my'])  # 188
	print(text_vocab.stoi['twists'])  # 930
	print(text_vocab.stoi['lets'])  # 956
	print(text_vocab.stoi['sports'])  # 1275
	print(text_vocab.stoi['amateurishly'])  # 6818


def test_method3(pad_index=0):
	batch_size = 2  # Only for demonstrative purposes
	shuffle = False  # Only for demonstrative purposes
	train_dataset = NLPDataset()
	train_dataset.from_file('data/sst_train_raw.csv')
	train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle,
								  collate_fn=pad_collate_fn)
	texts, labels, lengths = next(iter(train_dataloader))
	print(f"Texts: {texts}")
	print(f"Labels: {labels}")
	print(f"Lengths: {lengths}")


if __name__ == '__main__':
	# --- test1 ----
	dataset = NLPDataset()
	dataset.from_file('./data/sst_train_raw.csv')
	test_method1(dataset, dataset.text_vocab, dataset.label_vocab)

	# --- test2 ----
	# dataset = NLPDataset()
	# dataset.from_file('./data/sst_train_raw.csv')
	# freq = dataset.get_frequencies_examples()
	# test_method2(freq)

	# --- test3 ----
	# dataset = NLPDataset()
	# dataset.from_file('./data/sst_train_raw.csv')
	# freq = dataset.get_frequencies_labels()
	# print(freq)
	# tmp = Vocab(freq, label=True)
	# print(tmp.stoi, tmp.itos)

	# --- test4 ---
	# dataset = NLPDataset()
	# dataset.from_file('./data/sst_train_raw.csv')
	# m = dataset.text_vocab.get_embedding_matrix_random()
	# print(m.weight)
	# m = dataset.text_vocab.get_embedding_matrix_glove(path='./data/sst_glove_6b_300d.txt')
	# print(m.weight[143]) # 'here'

	# --- test5 ---
	# test_method3()
