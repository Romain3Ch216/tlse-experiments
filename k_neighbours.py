from TlseHypDataSet.tlse_hyp_data_set import TlseHypDataSet
from TlseHypDataSet.utils.dataset import DisjointDataSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from models.load_model import load_model
import numpy as np
import torch
import torch.nn as nn
import json
import pickle as pkl
import os
import pdb


dataset = TlseHypDataSet(
        '/scratchm/rthoreau/Datasets/Toulouse',
        pred_mode='pixel',
        low_level_only=True,
        patch_size=1,
        in_h5py=True,
        data_on_gpu=True,
    )

config = {
    #'model': 'spectral_vae',
    #'model': 'prototypical_net',
    'model': 'mae',
    #'model': 'AE',
    'batch_size': 256,
    'z_dim': 32,
    'conv_channels': 16,
    'device': 'cuda',
    'n_bands': dataset.n_bands,
    'n_classes': dataset.n_classes,
    'bbl': dataset.bbl,
}

splits = [DisjointDataSplit(dataset, splits=default_split) for default_split in dataset.default_splits]

# folder = '/scratchm/rthoreau/Experiments/spectral_vae/20230729-161843/'
# folder = '/scratchm/rthoreau/Experiments/prototypical_net/20230728-102203/'
# folder = '/scratchm/rthoreau/Experiments/prototypical_net/20230812-103718/'
# folder = '/scratchm/rthoreau/Experiments/pos_prototypical_net/20230809-090706/'
# folder = '/scratchm/rthoreau/Experiments/KNN/'
folder = '/scratchm/rthoreau/Experiments/mae/20230822-163608/'
# folder = '/scratchm/rthoreau/Experiments/AE/20230821-125036/'

# for split_id, split in enumerate(splits):
for split_id in range(3, 4):
    split = splits[split_id]

    split_id = split_id + 1
    with open(os.path.join(folder, f'split_{split_id}', 'config.json'), 'r') as f:
        config = json.load(f)

    config.update({
        'n_bands': dataset.n_bands,
        'n_classes': dataset.n_classes,
        'bbl': dataset.bbl
    })

    config['restore'] = None
    if 'cls_token' not in config:
        config['cls_token'] = False
    model = load_model(config)


    # checkpoint = torch.load('/scratchm/rthoreau/Experiments/spectral_vae/20230718-104817/split_1/best_model.pth.tar', map_location=config['device'])
    # checkpoint = torch.load('/scratchm/rthoreau/Experiments/strong_spectral_vae/20230727-134310/split_1/best_model.pth.tar', map_location=config['device'])
    # checkpoint = torch.load('/scratchm/rthoreau/Experiments/spectral_vae/20230718-104817/split_1/best_model.pth.tar', map_location=config['device'])
    checkpoint = torch.load(os.path.join(folder, f'split_{split_id}', 'best_model.pth.tar'), map_location=config['device'])

    checkpoint = checkpoint['state_dict']
    # checkpoint['positional_embedding.cached_penc'] = checkpoint['positional_embedding.cached_penc'][:1]
    # del checkpoint['positional_embedding.cached_penc']

    model.load_state_dict(checkpoint)
    # import pdb; pdb.set_trace()


    #for split_id, split in enumerate(splits):
    # for split_id in range(0, 1):
    #     split = splits[split_id]
    labeled_loader = torch.utils.data.DataLoader(split.sets_['labeled'], shuffle=True, batch_size=config['batch_size'], pin_memory=True)
    # unlabeled_loader = torch.utils.data.DataLoader(split.sets_['unlabeled'], shuffle=True, batch_size=config['batch_size'], pin_memory=True)
    val_loader = torch.utils.data.DataLoader(split.sets_['validation'], shuffle=False, batch_size=config['batch_size'], pin_memory=True)
    test_loader = torch.utils.data.DataLoader(split.sets_['test'], shuffle=False, batch_size=config['batch_size'], pin_memory=True)

    train_data = []
    train_labels = []

    # unlabeled_data = []

    val_data = []
    val_labels = []

    test_data = []
    test_labels = []

    for data, labels in labeled_loader:
        with torch.no_grad():
            # data, _ = model(data.view(data.shape[0], 1, 1, data.shape[-1]))
            #data, _ = model(data)
            if config['model'] == 'mae':
                data, _, _ = model.forward_encoder(data.view(data.shape[0], data.shape[-1]), mask_ratio=0)
                if model.is_cls_token:
                    data = data[:, 0, :]
                else:
                    data = torch.mean(data[:, 1:, :], dim=1)
            elif config['model'] == 'AE':
                data = model.latent(data.view(data.shape[0], data.shape[-1]))

        train_data.append(data.view(data.shape[0], -1))
        train_labels.append(labels.view(-1))

    for data, labels in val_loader:
        with torch.no_grad():
            # data, _ = model(data.view(data.shape[0], 1, 1, data.shape[-1]))
            #data, _ = model(data)
            if config['model'] == 'mae':
                data, _, _ = model.forward_encoder(data.view(data.shape[0], data.shape[-1]), mask_ratio=0)
                if model.is_cls_token:
                    data = data[:, 0, :]
                else:
                    data = torch.mean(data[:, 1:, :], dim=1)
            elif config['model'] == 'AE':
                data = model.latent(data.view(data.shape[0], data.shape[-1]))
        val_data.append(data.view(data.shape[0], -1))
        val_labels.append(labels.view(-1))

    for data, labels in test_loader:
        with torch.no_grad():
            # data, _ = model(data.view(data.shape[0], 1, 1, data.shape[-1]))
            #data, _ = model(data)
            if config['model'] == 'mae':
                data, _, _ = model.forward_encoder(data.view(data.shape[0], data.shape[-1]), mask_ratio=0)
                if model.is_cls_token:
                    data = data[:, 0, :]
                else:
                    data = torch.mean(data[:, 1:, :], dim=1)
            elif config['model'] == 'AE':
                data = model.latent(data.view(data.shape[0], data.shape[-1]))
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
    clf = KNeighborsClassifier(n_neighbors=3, leaf_size=30, p=2, weights='distance')
    """
    params = {
        'n_neighbors': [3, 5, 8],
        leaf_size': [20, 30, 40],
        'p': [1, 2],
        'weights': ['uniform', 'distance']
        }

    clf = RandomizedSearchCV(estimator=clf, param_distributions=params, cv=cv, n_iter=20, verbose=1)
    search = clf.fit(data, labels)
    # clf = cv_results['estimator'][0]
    cv_results = search.best_params_
    """
    clf.fit(data, labels)
    pred = clf.predict(test_data)
    OA = accuracy_score(pred, test_labels)
    F1 = f1_score(pred, test_labels, average=None)
    cm = confusion_matrix(test_labels, pred)

    test_metrics = {
         'accuracy': OA,
         'f1_score': list(F1)
        }

    np.save(os.path.join(folder, 'knn_split_{}_confusion_matrix.npy'.format(split_id)), cm)
    with open(os.path.join(folder, 'knn_split_{}_test_metrics.json'.format(split_id)), 'w') as f:
        json.dump(test_metrics, f, indent=4)

    with open(os.path.join(folder, 'knn_split_{}_clf.pkl'.format(split_id)), 'wb') as f:
        pkl.dump(clf, f)

    patch_pred = []
    for test_patch in dataset.test_patches:
        raster = dataset.image_rasters[test_patch['img_id']-1]
        coords = tuple([int(i) for i in test_patch['coords']])
        test_patch = raster.ReadAsArray(*coords, band_list=dataset.bands)
        test_patch = test_patch / 10**4
        test_patch = test_patch.transpose(1, 2, 0)
        shape = test_patch.shape[:2]
        test_patch = test_patch.reshape(-1, test_patch.shape[-1])
        test_pred = clf.predict(test_patch)
        patch_pred.append(test_pred.reshape(shape))

    with open(os.path.join(folder, f'split_{split_id}', 'test_patch_pred.pkl'), 'wb') as f:
        pkl.dump(patch_pred, f)
