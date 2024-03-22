from TlseHypDataSet.tlse_hyp_data_set import TlseHypDataSet
from TlseHypDataSet.utils.dataset import DisjointDataSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from models.load_model import load_model
import numpy as np
import torch
import torch.nn as nn
import json
import pickle as pkl
import os
import pdb
import sys
from tqdm import tqdm


dataset = TlseHypDataSet(
        '/path/to/dataset',
        pred_mode='pixel',
        patch_size=1,
        in_h5py=True,
        data_on_gpu=True,
    )

splits = [DisjointDataSplit(dataset, split=default_split) for default_split in dataset.standard_splits]

split_id = int(sys.argv[1])
split = splits[split_id - 1]


base_folder = '/path/to/dir'

batch_size = 1024
labeled_loader = torch.utils.data.DataLoader(split.sets_['train'], batch_size=batch_size, pin_memory=True)
val_loader = torch.utils.data.DataLoader(split.sets_['validation'], batch_size=batch_size, pin_memory=True)
test_loader = torch.utils.data.DataLoader(split.sets_['test'], batch_size=batch_size, pin_memory=True)

train_data = []
train_labels = []

val_data = []
val_labels = []

test_data = []
test_labels = []

for data, labels in tqdm(labeled_loader):
    data = data.view(data.shape[0], data.shape[-1])
    train_data.append(data.view(data.shape[0], -1))
    train_labels.append(labels.view(-1))

for data, labels in tqdm(val_loader):
    data = data.view(data.shape[0], data.shape[-1])
    val_data.append(data.view(data.shape[0], -1))
    val_labels.append(labels.view(-1))

for data, labels in tqdm(test_loader):
    data = data.view(data.shape[0], data.shape[-1])
    test_data.append(data.view(data.shape[0], -1))
    test_labels.append(labels.view(-1))

train_data = torch.cat(train_data, dim=0).numpy()
train_labels = torch.cat(train_labels, dim=0).numpy()

val_data = torch.cat(val_data, dim=0).numpy()
val_labels = torch.cat(val_labels, dim=0).numpy()

test_data = torch.cat(test_data, dim=0).numpy()
test_labels = torch.cat(test_labels, dim=0).numpy()

data = np.concatenate((train_data, val_data), axis=0)
labels = np.concatenate((train_labels, val_labels), axis=0)
train_indices = np.arange(len(train_labels))
val_indices = np.arange(len(train_labels), len(train_labels) + len(val_labels))

cv = [(train_indices, val_indices)]

clf = SVC(kernel='rbf', class_weight='balanced', C=1, verbose=True)
search = clf.fit(data, labels)
pred = clf.predict(test_data)

OA = accuracy_score(pred, test_labels)
F1 = f1_score(pred, test_labels, average=None)
avg_F1 = f1_score(pred, test_labels, average='macro')

test_metrics = {
    'OA': OA,
    'f1_score': list(F1),
    'avg F1': avg_F1
}

with open(os.path.join(base_folder, 'SVM_split_{}_test_metrics.json'.format(split_id)), 'w') as f:
    json.dump(test_metrics, f, indent=4)
