import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
from zadatak1 import *

max_size = -1
min_freq = 1
seed = 7052020
lr = 1e-4
train_batch_size = 10
validate_batch_size = 32
test_batch_size = 32

class MyRNN(nn.Module):
	def __init__(self, embedding_matrix):
		super(MyRNN, self).__init__()
		self.rnn1 = nn.RNN(input_size=300, hidden_size=150)
		self.rnn2 = nn.RNN(input_size=150, hidden_size=150)
		self.fc1 = nn.Linear(150, 150)
		self.fc2 = nn.Linear(150, 1)
		self.embedding_matrix = embedding_matrix

	def forward(self, x):
		x = self.embedding_matrix(x)
		x, _ = self.rnn1(x)
		x, _ = self.rnn2(x)
		x = self.fc1(x[:, -1, :])
		x = F.relu(x)
		x = self.fc2(x)
		return x.reshape(-1)

def initialize_model(embedding_matrix):
	return MyRNN(embedding_matrix)

def train(model, data, optimizer, criterion):
	eval_loss = []
	cm = 0
	correct = 0
	count = 0
	model.train()
	for batch_num, batch in enumerate(data):
		model.zero_grad()
		x, y, l = batch
		logits = model(x)

		y = torch.stack(y, dim=0)
		y_ = (torch.sigmoid(logits) >= 0.5).int()
		loss = criterion(logits, y.float())
		optimizer.zero_grad()
		loss.backward()
		# torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
		optimizer.step()

		cm += confusion_matrix(y_true=y, y_pred=y_)
		eval_loss.append(loss)

	eval_loss = torch.mean(torch.tensor(eval_loss))
	# eval_acc = correct / count
	calculate_metrics(cm, eval_loss)


def evaluate(model, data, criterion):
	cm = 0
	model.eval()
	with torch.no_grad():
		eval_loss = []
		correct = 0
		count = 0
		for batch_num, batch in enumerate(data):
			x, y, l = batch
			logits = model(x)

			y = torch.stack(y, dim=0)
			y_ = (torch.sigmoid(logits) >= 0.5).int()

			loss = criterion(logits, y.float())
			eval_loss.append(loss)

			correct += ((torch.sigmoid(logits) >= 0.5).float() == y).float().sum()
			count += len(y)

			cm += confusion_matrix(y_true=y, y_pred=y_)

	eval_loss = torch.mean(torch.tensor(eval_loss))
	eval_acc = correct / count
	calculate_metrics(cm, eval_loss)

	return eval_acc

def load_dataset():
	train_dataset = NLPDataset()
	train_dataset.from_file("./data/sst_train_raw.csv")

	valid_dataset = NLPDataset()
	valid_dataset.from_file("./data/sst_valid_raw.csv")

	test_dataset = NLPDataset()
	test_dataset.from_file("./data/sst_test_raw.csv")

	return train_dataset, valid_dataset, test_dataset

def calculate_metrics(cm, eval_loss):
	tn = cm[0][0]
	tp = cm[1][1]
	fp = cm[0][1]
	fn = cm[1][0]
	sensitivity = tp / (tp + fn)
	precision = tp / (tp + fp)

	print('Confusion Matirx : ')
	print(cm)
	print('- Sensitivity : ', (tp / (tp + fn)) * 100)
	print('- Specificity : ', (tn / (tn + fp)) * 100)
	print('- Precision: ', (tp / (tp + fp)) * 100)
	print('- NPV: ', (tn / (tn + fn)) * 100)
	print('- F1: ', ((2 * sensitivity * precision) / (sensitivity + precision)) * 100)
	print('- Loss: ', eval_loss.item())
	print()

	return cm

if __name__ == '__main__':
	np.random.seed(seed)
	torch.manual_seed(seed)

	epochs = 5

	train_dataset, valid_dataset, test_dataset = load_dataset()
	model = initialize_model(train_dataset.text_vocab.get_embedding_matrx())

	criterion = nn.BCEWithLogitsLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=pad_collate_fn)
	validation_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=validate_batch_size, shuffle=False, collate_fn=pad_collate_fn)
	testing_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=pad_collate_fn)

	for epoch in range(epochs):
		train(model, training_loader, optimizer, criterion)
		accuracy = evaluate(model, validation_loader, criterion)
		print(f"Epoch {epoch}: valid accuracy = {accuracy}")
	accuracy = evaluate(model, testing_loader, criterion)
	print(f"Test accuracy = {accuracy}")
