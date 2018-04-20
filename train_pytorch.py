#!/usr/bin/env python
import argparse
import os
import sys
import numpy as np
from text_utils import char2vec, n_chars
from glob import glob
from tqdm import tqdm,trange

import torch
import torch.nn  as nn
import torch.optim as optim
from torch.autograd import Variable

dir_a = 'data/sklearn_clean/'
dir_b = 'data/scalaz_clean/'
train_a = glob(os.path.join(dir_a, "train/*"))
train_b = glob(os.path.join(dir_b, "train/*"))
val_a = glob(os.path.join(dir_a, "test/*"))
val_b = glob(os.path.join(dir_b, "test/*"))
seq_len = 100
ngen = 1024
jump_size_a = [20, 200]
jump_size_b = [20, 200]
min_jump_size_a = 20
max_jump_size_a = 200
min_jump_size_b = 20
max_jump_size_b = 200
batch_size = 1024
seq_len = 100
juma = [min_jump_size_a, max_jump_size_a]
jumb = [min_jump_size_b, max_jump_size_b]


def chars_from_files(list_of_files):
    while True:
        filename = choice(list_of_files)
        with open(filename, 'r') as f:
            chars = f.read()
            for c in chars:
                yield c

def splice_texts(files_a, jump_size_a, files_b, jump_size_b):
    a_chars = chars_from_files(files_a)
    b_chars = chars_from_files(files_b)
    generators = [a_chars, b_chars]

    a_range = range(jump_size_a[0], jump_size_a[1])
    b_range = range(jump_size_b[0], jump_size_b[1])
    ranges = [a_range, b_range]

    source_ind = choice([0, 1])
    while True:
        jump_size = choice(ranges[source_ind])
        gen = generators[source_ind]
        for _ in range(jump_size):
            yield (gen.__next__(), source_ind)
        source_ind = 1 - source_ind

use_cuda = torch.cuda.is_available()

class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.contiguous().view(t * n, -1)
        x = self.module(x)
        x = x.contiguous().view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


def generate_batches(files_a, jump_size_a, files_b, jump_size_b, batch_size, sample_len, return_text=False):
    # This generators will generate an infinite sequence of characters
    # belonging to source 1/0. per bach, the sequence will be "cut" as
    # sample_len
    gens = [splice_texts(files_a, jump_size_a, files_b, jump_size_b) for _ in range(batch_size)]
    while True:
        X = []
        y = []
        texts = []
        for g in gens:
            chars = []
            vecs = []
            labels = []
            for _ in range(sample_len):
                c, l = g.__next__()
                vecs.append(char2vec[c])
                labels.append([l])
                chars.append(c)
            X.append(vecs)
            y.append(labels)

            if return_text:
                texts.append(''.join(chars))

        if return_text:
            yield (np.array(X), np.array(y), texts)
        else:
            yield (np.array(X), np.array(y))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self,y_pred,y):
        y_pred = (y_pred.view(-1, 1) > 0.5).data.float()
        y = y.view(-1, 1).data.float()
        acc = (y_pred == y).sum()/y.size(0)
        return Variable(torch.FloatTensor([acc]))

    def __repr__(self):
        return self.__class__.__name__ + '(\n)'


class RNNCharTagger(nn.Module):
    def __init__(self, lstm_layers, input_dim, out_dim, batch_size=1024, dropout=0.2, batch_first=True):
        super(RNNCharTagger, self).__init__()

        self.lstm_layers = lstm_layers
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.batch_first = batch_first
        self.batch_size = batch_size

        self.lstm1 =  nn.LSTM(self.input_dim, self.out_dim, batch_first=self.batch_first, dropout=self.dropout)
        for i in range(1,self.lstm_layers):
            setattr(self, 'lstm'+str(i+1), nn.LSTM(self.out_dim, self.out_dim, batch_first=self.batch_first, dropout=self.dropout))
        self.linear = SequenceWise(nn.Linear(self.out_dim, 1))

        for i in range(self.lstm_layers):
            # setattr(self, 'h'+str(i+1), nn.Parameter(torch.zeros(1, self.batch_size, self.out_dim)))
            # setattr(self, 'c'+str(i+1), nn.Parameter(torch.zeros(1, self.batch_size, self.out_dim)))
            setattr(self, 'h'+str(i+1), nn.Parameter(nn.init.normal(torch.Tensor(1, self.batch_size, self.out_dim))))
            setattr(self, 'c'+str(i+1), nn.Parameter(nn.init.normal(torch.Tensor(1, self.batch_size, self.out_dim))))

    def forward(self, X):

        output, (h1, c1) = self.lstm1(X, (self.h1, self.c1))
        hidden_states = [(h1,c1)]
        for i in range(1,self.lstm_layers):
            h,c = getattr(self, 'h'+str(i+1)), getattr(self, 'c'+str(i+1))
            output, (nh,nc) = getattr(self, 'lstm'+str(i+1))(output, (h,c))
            hidden_states.append((nh,nc))

        for i in range(self.lstm_layers):
            setattr(self, 'h'+str(i+1), nn.Parameter(hidden_states[i][0].data))
            setattr(self, 'c'+str(i+1), nn.Parameter(hidden_states[i][1].data))

        output = F.sigmoid(self.linear(output))

        return output


class BiRNNCharTagger(nn.Module):
    def __init__(self, lstm_layers, input_dim, out_dim, batch_size=1024, dropout=0.2, batch_first=True):
        super(BiRNNCharTagger, self).__init__()

        self.lstm_layers = lstm_layers
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.batch_first = batch_first
        self.batch_size = batch_size

        self.lstm =  nn.LSTM(
            self.input_dim,
            self.out_dim,
            batch_first=self.batch_first,
            dropout=self.dropout,
            num_layers = self.lstm_layers,
            bidirectional=True)
        self.linear = SequenceWise(nn.Linear(2*self.out_dim, 1))

    def forward(self, X):
        lstm_output, hidden = self.lstm(X)
        output = F.sigmoid(self.linear(lstm_output))
        return output


model = BiRNNCharTagger(3,96,128)
model = model.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
train_gen = generate_batches(train_a, juma, train_b, jumb, batch_size, seq_len)
val_gen = generate_batches(val_a, juma, val_b, jumb, batch_size, seq_len)
epochs = 3
steps_per_epoch = 100
validation_steps = 50

def train(train_gen, model, criterion, optimizer, epoch, steps_per_epoch):

    # switch to train mode
    model.train()

    with trange(steps_per_epoch) as t:
        for i in t:
            t.set_description('epoch %i' % epoch)
            X,y = train_gen.__next__()
            X_var = Variable(torch.from_numpy(X).float())
            y_var = Variable(torch.from_numpy(y).float())
            if use_cuda:
                X_var, y_var = X_var.cuda(), y_var.cuda()
            optimizer.zero_grad()
            y_pred = model(X_var)
            loss = criterion(y_pred, y_var)
            t.set_postfix(loss=loss.data[0])
            loss.backward()
            optimizer.step()

def validate(val_gen, model, metrics, validation_steps):

    # switch to evaluate mode
    model.eval()

    losses = []
    for i in range(len(metrics)):
        losses.append(AverageMeter())

    with trange(validation_steps) as t:
        for i in t:
            t.set_description('validating')
            X,y = val_gen.__next__()
            X_var = Variable(torch.from_numpy(X).float())
            y_var = Variable(torch.from_numpy(y).float())
            if use_cuda:
                X_var, y_var = X_var.cuda(), y_var.cuda()
            y_pred = model(X_var)
            for i in range(len(metrics)):
                losses[i].update(metrics[i](y_pred, y_var).data[0])

        for metric,loss in zip(metrics, losses):
            print("val_{}: {}".format(metric.__repr__().split("(")[0], loss.val))


metrics = [nn.MSELoss(), nn.BCELoss(), Accuracy()]
for epoch in range(epochs):
    train(train_gen, model, criterion, optimizer, epoch, steps_per_epoch)
    validate(val_gen, model, metrics, validation_steps)


MODEL_DIR = 'models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
torch.save(model.state_dict(), os.path.join(MODEL_DIR,'model.pkl'))

###############################################################################