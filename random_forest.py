from TlseHypDataSet.tlse_hyp_data_set import TlseHypDataSet
from TlseHypDataSet.utils.dataset import DisjointDataSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
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

base_folder = sys.argv[1]
splits = [DisjointDataSplit(dataset, split=default_split) for default_split in dataset.standard_splits]
batch_size = 1024

split_id = int(sys.argv[2])
split = splits[split_id - 1]

raw = sys.argv[3] == 'raw'

if raw:
    config = {'model': 'raw'}
    base_folder = '/path/to/exp',
else:
    folder = os.path.join(base_folder, 'split_{}'.format(split_id))

    with open(os.path.join(folder, 'config.json'), 'r') as f:
        config = json.load(f)

    config['device'] = 'cpu'
    model = load_model(config)
    model.to(config['device'])
    model.eval()

    checkpoint = torch.load(os.path.join(folder, 'best_model.pth.tar'), map_location=config['device'])
    checkpoint = checkpoint['state_dict']
    model.load_state_dict(checkpoint)

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
    with torch.no_grad():
        if config['model'] == 'MAE':
            _, _, _, data = model.forward(data.view(data.shape[0], data.shape[-1]), mask_ratio=0)
        elif config['model'] == 'AE':
            data, _ = model.forward(data.view(data.shape[0], data.shape[-1]))
        else:
            data = data.view(data.shape[0], data.shape[-1])
    train_data.append(data.view(data.shape[0], -1))
    train_labels.append(labels.view(-1))

for data, labels in tqdm(val_loader):
    with torch.no_grad():
        if config['model'] == 'MAE':
            _, _, _, data = model.forward(data.view(data.shape[0], data.shape[-1]), mask_ratio=0)
        elif config['model'] == 'AE':
            data, _ = model.forward(data.view(data.shape[0], data.shape[-1]))
        else:
            data = data.view(data.shape[0], data.shape[-1])
    val_data.append(data.view(data.shape[0], -1))
    val_labels.append(labels.view(-1))

for data, labels in tqdm(test_loader):
    with torch.no_grad():
        if config['model'] == 'MAE':
            _, _, _, data = model.forward(data.view(data.shape[0], data.shape[-1]), mask_ratio=0)
        elif config['model'] == 'AE':
            data, _ = model.forward(data.view(data.shape[0], data.shape[-1]))
        else:
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

estimator = RandomForestClassifier(class_weight='balanced')
params = {
    'n_estimators': [400, 800, 1600, 2400],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [2, 4],
    'max_depth': [20, 50 ,70, 100],
    'max_features': ['log2', 'sqrt'],
}
clf = RandomizedSearchCV(estimator=estimator, param_distributions=params, cv=cv, n_iter=20, verbose=1)
search = clf.fit(data, labels)
print(search.best_params_)
pred = clf.predict(test_data)

OA = accuracy_score(pred, test_labels)
F1 = f1_score(pred, test_labels, average=None)
avg_F1 = f1_score(pred, test_labels, average='macro')

test_metrics = {
    'OA': OA,
    'f1_score': list(F1),
    'avg F1': avg_F1
}

with open(os.path.join(base_folder, 'RF_split_{}_test_metrics.json'.format(split_id)), 'w') as f:
    json.dump(test_metrics, f, indent=4)
