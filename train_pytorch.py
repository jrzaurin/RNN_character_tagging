#!/usr/bin/env python
import argparse
import os
import sys
from glob import glob
from random import choice
import numpy as np
from text_utils import char2vec, n_chars
from tqdm import tqdm,trange

import torch
import torch.nn  as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


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


class CharRNNLoader(Dataset):

    def __init__(self, seq_len, char2vec, gen, ngen, train_a, train_b, jump_size_a,
        jump_size_b, return_text=False):

        self.seq_len = seq_len
        self.char2vec = char2vec
        self.return_text = return_text
        self.gens = [gen(train_a, jump_size_a, train_b, jump_size_b) for _ in range(ngen)]

    def __getitem__(self, idx):

        X = []
        y = []
        texts = []

        for g in self.gens:
            chars = []
            vecs = []
            labels = []
            for _ in range(self.seq_len):
                c, l = g.__next__()
                vecs.append(self.char2vec[c])
                labels.append([l])
                chars.append(c)
            X.append(vecs)
            y.append(labels)
            if self.return_text:
                texts.append(''.join(chars))

        if self.return_text:
            return np.array(X), np.array(y), texts
        else:
            return np.array(X), np.array(y)

    def __len__(self):
        return sys.maxsize
        # return 5

# chardataset = CharRNNLoader(seq_len, char2vec, splice_texts, 1024, train_a, train_b, jump_size_a, jump_size_b)
# train_loader = DataLoader(dataset=chardataset, batch_size=1, shuffle=False)

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


# class RNNCharTagger(nn.Module):
#     def __init__(self):
#         super(RNNCharTagger, self).__init__()

#         self.lstm1 = nn.LSTM(96, 128, batch_first=True, dropout=0.2)
#         self.lstm2 = nn.LSTM(128, 128, batch_first=True, dropout=0.2)
#         self.lstm3 = nn.LSTM(128, 128, batch_first=True, dropout=0.2)
#         self.linear = SequenceWise(nn.Linear(128, 1))

#         self.h1 = nn.Parameter(torch.zeros(1, 1024, 128))
#         self.c1 = nn.Parameter(torch.zeros(1, 1024, 128))
#         self.h2 = nn.Parameter(torch.zeros(1, 1024, 128))
#         self.c2 = nn.Parameter(torch.zeros(1, 1024, 128))
#         self.h3 = nn.Parameter(torch.zeros(1, 1024, 128))
#         self.c3 = nn.Parameter(torch.zeros(1, 1024, 128))

#     def forward(self, X):

#         output1, (h1, c1) = self.lstm1(X,       (self.h1, self.c1))
#         output2, (h2, c2) = self.lstm2(output1, (self.h2, self.c2))
#         output3, (h3, c3) = self.lstm3(output2, (self.h3, self.c3))

#         self.h1 = nn.Parameter(h1.data)
#         self.c1 = nn.Parameter(c1.data)
#         self.h2 = nn.Parameter(h2.data)
#         self.c2 = nn.Parameter(c2.data)
#         self.h3 = nn.Parameter(h3.data)
#         self.c3 = nn.Parameter(c3.data)

#         output = F.sigmoid(self.linear(output3))

#         return output


class RNNCharTagger(nn.Module):
    def __init__(self, lstm_layers, input_dim, out_dim, batch_size=1024,dropout=0.2, batch_first=True):
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
            setattr(self, 'h'+str(i+1), nn.Parameter(torch.zeros(1, self.batch_size, self.out_dim)))
            setattr(self, 'c'+str(i+1), nn.Parameter(torch.zeros(1, self.batch_size, self.out_dim)))

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

model = RNNCharTagger(3,96,128)
model = model.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
train_gen = generate_batches(train_a, juma, train_b, jumb, batch_size, seq_len)

epochs = 2
steps_per_epoch = 100
for epoch in range(epochs):
    with trange(steps_per_epoch) as t:
        for i in t:
            t.set_description('epoch %i' % epoch)
            X,y = train_gen.__next__()
            inp, targets = Variable(torch.from_numpy(X).float()), Variable(torch.from_numpy(y).float())
            inp, targets = inp.cuda(), targets.cuda()
            model.zero_grad()
            # model.hidden = model.init_hidden()
            scores = model(inp)
            loss = criterion(scores, targets)
            t.set_postfix(loss=loss.data[0])
            loss.backward()
            optimizer.step()


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

model = RNNCharTagger(3,96,128)
model = model.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
train_gen = generate_batches(train_a, juma, train_b, jumb, batch_size, seq_len)
epochs = 2
steps_per_epoch = 100

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
            # model.hidden = model.init_hidden()
            y_pred = model(X_var)
            loss = criterion(y_pred, y_var)
            t.set_postfix(loss=loss.data[0])
            loss.backward()
            optimizer.step()


for epoch in range(epochs):
    train(train_gen, model, criterion, optimizer, epoch, steps_per_epoch)

def validate(val_gen, model, metrics, validation_steps):

    # switch to evaluate mode
    model.eval()
    if isinstance(metrics, list):
        losses = []
        for i in range(len(metrics)):
            losses.append(AverageMeter())

    with trange(validation_steps) as t:
        for i in t:
            X,y = val_gen.__next__()
            X_var = Variable(torch.from_numpy(X).float())
            y_var = Variable(torch.from_numpy(y).float())
            if use_cuda:
                X_var, y_var = X_var.cuda(), y_var.cuda()
            y_pred = model(X_var)
            if isinstance(metrics, list):
                for i in range(len(metrics)):
                    losses[i].update(metrics[i](y_pred, y_var).data[0])
            else:
                loss = metrics(y_pred,y_var)

    for metric,loss in zip(metrics, losses):
        print("val_{}: {}".format(metric.__repr__().split("(")[0], loss.val))





###############################################################################








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


def main(model_path, dir_a, dir_b, min_jump_size_a, max_jump_size_a, min_jump_size_b,
    max_jump_size_b, seq_len, batch_size, rnn_size, lstm_layers, dropout_rate,
    bidirectional, steps_per_epoch, epochs):

        train_a = glob(os.path.join(dir_a, "train/*"))
        train_b = glob(os.path.join(dir_b, "train/*"))
        val_a = glob(os.path.join(dir_a, "test/*"))
        val_b = glob(os.path.join(dir_b, "test/*"))

        juma = [min_jump_size_a, max_jump_size_a]
        jumb = [min_jump_size_b, max_jump_size_b]
        batch_shape = (batch_size, seq_len, n_chars)
        if os.path.isfile(model_path):
            model = load_model(model_path)
            batch_size, seq_len, _ = model.input_shape
        else:
            model = Sequential()
            for _ in range(lstm_layers):
                if bidirectional:
                    model.add(Bidirectional(LSTM(rnn_size, return_sequences=True),
                                            batch_input_shape=batch_shape))
                else:
                    model.add(LSTM(rnn_size, return_sequences=True, batch_input_shape=batch_shape,
                                   stateful=True))

                model.add(Dropout(dropout_rate))

            model.add(TimeDistributed(Dense(units=1, activation='sigmoid')))
            model.compile(optimizer='adam', loss='mse', metrics=['accuracy', 'binary_crossentropy'])

        train_gen = generate_batches(train_a, juma, train_b, jumb, batch_size, seq_len)
        validation_gen = generate_batches(val_a, juma, val_b, jumb, batch_size, seq_len)
        checkpointer = ModelCheckpoint(model_path)

        model.fit_generator(train_gen,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=validation_gen,
                            validation_steps=100,
                            epochs=epochs,
                            callbacks=[checkpointer])