import argparse
import os
import sys
import numpy as np

from joblib import dump
from text_utils import char2vec, n_chars
from random import choice
from glob import glob
from tqdm import tqdm,trange

import torch
import torch.nn  as nn
import torch.optim as optim
from torch.autograd import Variable
from torch_utils import AverageMeter, Accuracy, RNNCharTagger, BiRNNCharTagger

dir_a = 'data/sklearn_clean/'
dir_b = 'data/scalaz_clean/'
# dir_a = 'data/austen_clean/'
# dir_b = 'data/shakespeare_clean/'
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


def generate_batches(files_a, jump_size_a, files_b, jump_size_b, batch_size, sample_len, return_text=False):
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


use_cuda = torch.cuda.is_available()
model  = BiRNNCharTagger(3,96,128)
model.load_state_dict(torch.load('models/model.pkl'))
model = model.cuda()
model.eval()

gen = generate_batches(val_a, juma, val_b, jumb, batch_size, seq_len, return_text=True)
steps = 50
predictions, labels, texts = [],[],[]
with trange(steps) as t:
    for i in t:
        X,y,text = gen.__next__()
        X_var = Variable(torch.from_numpy(X).float())
        y_var = Variable(torch.from_numpy(y).float())
        if use_cuda:
            X_var, y_var = X_var.cuda(), y_var.cuda()
        pr = model(X_var)
        predictions.append(pr.data)
        labels.append(y_var.data)
        texts.append(text)

preds = torch.cat(predictions,dim=1).reshape(batch_size,steps*seq_len)
preds = preds.cpu().numpy()
labs = torch.cat(labels,dim=1).reshape(batch_size,steps*seq_len)
labs = labs.cpu().numpy()
txts = []
for j in range(1024):
    txts.append("".join([texts[i][j] for i in range(50)]))

output_path = 'sklearn_clean_pr'
try:
    os.makedirs(output_path)
except os.error:
    pass
for i in range(batch_size):
    path = os.path.join(output_path, 'part_' + str(i).zfill(5) + ".joblib")
    dump((txts[i], preds[i], labs[i]), path)