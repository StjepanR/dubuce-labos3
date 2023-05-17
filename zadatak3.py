import torch.nn as nn

from sklearn.metrics import confusion_matrix
from zadatak1 import *

max_size = -1
min_freq = 1
seed = 7052020
lr = 1e-4
train_batch_size = 10
validate_batch_size = 32
test_batch_size = 32
gradient_clip = 0.25

class MyRNN(nn.Module):
	def __init__(self, embedding_matrix):
		super(MyRNN, self).__init__()
		self.rnn1 = nn.RNN(input_size=300, hidden_size=150, num_layers=2)
		self.fc1 = nn.Linear(150, 150)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(150, 1)
		self.embedding_matrix = embedding_matrix

	def forward(self, x):
		x = self.embedding_matrix(x)
		x = torch.transpose(x, 0, 1)
		_, h_n = self.rnn1(x)
		out = self.fc1(h_n[-1])
		out = self.relu(out)
		out = self.fc2(out)
		return out.reshape(-1)

def initialize_model(embedding_matrix):
	return MyRNN(embedding_matrix)

def train(model, data, optimizer, criterion):
	cm = 0
	model.train()
	for batch_num, batch in enumerate(data):
		# model.zero_grad()
		# Every data instance is an input + label pair
		x, y, _ = batch
		y = torch.stack(y, dim=0)

		# Zero your gradients for every batch!
		optimizer.zero_grad()

		# Make predictions for this batch
		logits = model(x)

		# Compute the loss and its gradients
		loss = criterion(logits, y.float())
		loss.backward()

		# torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

		y_ = (logits > 0.).int()

		# Adjust learning weights
		optimizer.step()

		cm += confusion_matrix(y_true=y, y_pred=y_)

	calculate_metrics(cm, None)


def evaluate(model, data, criterion):
	cm = 0
	model.eval()
	with torch.no_grad():
		eval_loss = []
		correct = 0
		count = 0
		for batch_num, batch in enumerate(data):
			x, y, l = batch
			y = torch.stack(y, dim=0)

			logits = model(x)

			loss = criterion(logits, y.float())
			eval_loss.append(loss.item())

			y_ = (logits > 0.).int()

			correct += (y_ == y).sum()
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
	valid_dataset.text_vocab = train_dataset.text_vocab
	valid_dataset.label_vocab = train_dataset.label_vocab

	test_dataset = NLPDataset()
	test_dataset.from_file("./data/sst_test_raw.csv")
	test_dataset.text_vocab = train_dataset.text_vocab
	test_dataset.label_vocab = train_dataset.label_vocab

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
	if eval_loss is not None:
		print('- Loss: ', eval_loss.item())
	print()

	return cm

if __name__ == '__main__':
	epochs = 5

	train_dataset, valid_dataset, test_dataset = load_dataset()
	model = initialize_model(train_dataset.text_vocab.get_embedding_matrix())

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
