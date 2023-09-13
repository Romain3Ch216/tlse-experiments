# from learning.trainer import AutoEncoding, Classification
# from models.autoencoder import AutoEncoder
# from models.mlp import MLP
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import cycle
from torch.autograd import Variable
import numpy as np

def load_config(config):
    if 'model' not in config or config['model'] in ['hsi', '1d_autoencoder', 'p3VAE', 'sp3VAE', 'gaussianVAE',
                                                    'mlp', 'hu', 'prototypical_net', 'strong_spectral_vae',
                                                    'rec_spectral_vae', 'spectral_vae', 'linear_head',
                                                    'pos_prototypical_net', 'mae', 'AE']:
        config.update({
            'pred_mode': 'pixel',
            'low_level_only': True,
            'patch_size': 1
        })
    elif config['model'] in ['deep_net_segmentation', 'rfg_unet', 'fg_unet_urban',  'fg_unet', 'pretrained_fg_unet',
                             'lsu', 'spectral_unet', 'deep_net', 'deep_net_kmeans', 'shallow_net', 'fg_unet_supervised',
                             'mlp', 'siamese_net', 'gabor_net']:
        config.update({
            'pred_mode': 'patch',
            'low_level_only': True,
            'patch_size': 32
        })
    elif config['model'] in ['patch_spectral_vae', 'spectral_net']:
        config.update({
            'pred_mode': 'pixel',
            'low_level_only': True,
            'patch_size': 1
        })
    elif config['model'] == 'li':
        config.update({
            'pred_mode': 'pixel',
            'low_level_only': True,
            'patch_size': 5
        })
    return config

"""
def load_model(config, dataset):
    if config['model'] == '1d_autoencoder':
        model = AutoEncoder(dataset.n_bands, dataset.bbl, 16, config['h_dim'], z_dim=16, dropout=0.5)
        trainer = AutoEncoding(dataset, model, config)
    elif config['model'] == 'mlp':
        model = MLP(dataset.n_bands, config['h_dim'], dataset.n_classes, config['dropout'])
        trainer = Classification(dataset, model, config)
    return trainer
"""
def update_config(config, study):
    for k, v in study.best_params.items():
        config[k] = v
    config['seed'] = study.best_trial.user_attrs['seed']
    return config

def sam_(x, y, reduction='mean'):
    """
    Calculates the spectral angle between two batches of vectors.
    """
    x_norm = 1e-6 + torch.linalg.norm(x, dim=-1)
    y_norm = 1e-6 + torch.linalg.norm(y, dim=-1)
    dot_product = torch.bmm(x.unsqueeze(1), y.unsqueeze(-1))
    prod = dot_product.view(-1)/(x_norm*y_norm)
    prod = torch.clamp(prod, 1e-6, 1-1e-6)
    assert all(prod >= torch.zeros_like(prod)) and all(prod <= torch.ones_like(prod)), "Out of [0,1]"
    sam = torch.acos(prod)
    assert not torch.isnan(sam.mean()), "SAM contains NaN"
    if reduction == 'mean':
        return sam.mean()
    elif reduction == 'none':
        return sam

def dist_sam(X, Y):
    dot_product = torch.mm(X, Y.T)
    x_norm = 1e-6 + torch.linalg.norm(X, dim=-1)
    y_norm = 1e-6 + torch.linalg.norm(Y, dim=-1)
    prod = dot_product/x_norm.unsqueeze(1)/y_norm.unsqueeze(0)
    prod = torch.clamp(prod, 1e-6, 1-1e-6)
    sam = torch.acos(prod)
    return sam


def Lr_(u, v, alpha=1, weights=None):
    if weights is None:
        mse = F.mse_loss(u, v)
        sam = sam_(u, v)
    else:
        mse = F.mse_loss(u, v, reduction='none')
        mse = torch.mean(mse, dim=-1)*weights
        sam = sam_(u, v, reduction='none')*weights
    return mse + sam


def one_hot(y, y_dim):
    """
    Returns labels in a one-hot format.
    """
    batch_size = len(y)
    one_hot = torch.zeros(batch_size, y_dim)
    one_hot[torch.arange(batch_size), y] = 1
    return one_hot


def enumerate_discrete(x, y_dim):
    """
    Generates a `torch.Tensor` of size batch_size x n_labels of
    the given label.

    Example: generate_label(2, 1, 3) #=> torch.Tensor([[0, 1, 0],
                                                       [0, 1, 0]])
    :param x: tensor with batch size to mimic
    :param y_dim: number of total labels
    :return variable
    """
    def batch(batch_size, label):
        labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
        y = torch.zeros((batch_size, y_dim))
        y.scatter_(1, labels, 1)
        return y.type(torch.LongTensor)

    batch_size = x.size(0)
    generated = torch.cat([batch(batch_size, i) for i in range(y_dim)])

    if x.is_cuda:
        generated = generated.cuda()

    return Variable(generated.float())

def cycle_loader(labeled_data, unlabeled_data):
    return zip(cycle(labeled_data), unlabeled_data)


class PrototypicalBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, classes_per_it, num_samples, iterations):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        '''
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.sample_per_class = self.counts.min()
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        cpi = self.classes_per_it

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.zeros(batch_size).long()
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[c_idxs]):
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                batch[i * spc: (i + 1) * spc] = self.indexes[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations

def init_sampler(labels, iterations, classes_per_it, num_samples):
    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=iterations)


def init_dataloader(dataset, labels, iterations, classes_per_it, num_samples):
    sampler = init_sampler(labels, iterations, classes_per_it, num_samples)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader

class PrototypicalTestSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, num_samples, iterations):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        '''
        super(PrototypicalTestSampler, self).__init__()
        self.labels = labels
        self.sample_per_class = num_samples
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)
        self.n_classes = len(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        for it in range(self.iterations):
            batch = torch.zeros((self.n_classes, spc)).long()
            for i in range(len(self.classes)):
                sample_idxs = torch.randperm(self.numel_per_class[i])[:spc]
                batch[i, :] = self.indexes[i][sample_idxs]
            yield batch

def init_test_sampler(labels, iterations, num_samples):
    return PrototypicalTestSampler(labels=labels,
                                    num_samples=num_samples,
                                    iterations=iterations)
