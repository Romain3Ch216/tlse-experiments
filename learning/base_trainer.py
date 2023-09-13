import torch
import torch.nn as nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import json
import os
import collections
from learning.knn import knn_pred


class Trainer(object):
    def __init__(self, dataset, model, config):
        self.optimizer = None
        self.criterion = None
        self.metrics = None
        self.device = None
        self.model = model
        self.dataset = dataset

        assert 'log_dir' in config
        assert 'epochs' in config

        self.config = config
        self.logger = SummaryWriter(logdir=config['log_dir'])
        self.target_metric = None
        self.direction = None
        self.best_metric = None
        self.last_metrics = collections.deque(maxlen=config['patience'])
        self.stop_training = False
        self.save_best_model = config['save_best_model']

        config_to_file = {}
        for k, v in config.items():
            if isinstance(v, np.int64):
                config_to_file[k] = int(v)
            elif isinstance(v, np.ndarray):
                config_to_file[k] = list(v)
            elif isinstance(v, torch.Tensor):
                config_to_file[k] = None
            elif isinstance(v, bool):
                config_to_file[k] = str(v)
            elif v is not None:
                config_to_file[k] = v
        if 'bbl' in config_to_file:
            del config_to_file['bbl']

        with open(os.path.join(config['log_dir'], 'config.json'), 'w') as f:
            json.dump(config_to_file, f, indent=4)

        reset_parameters(self.model, config['seed'])

    def reset_meters(self, epoch):
        val_metric = self.meters['val'][self.target_metric].value()[0]

        if self.direction * val_metric <= self.direction * self.best_metric:
            self.best_metric = val_metric
            if self.save_best_model:
                torch.save(
                    {'epoch': epoch,
                     'best_metric': self.best_metric,\
                     'state_dict': self.model.state_dict()}, os.path.join(self.config['log_dir'], 'best_model.pth.tar')
                )

        for set_ in ['train', 'val']:
            for meter in self.meters[set_]:
                self.logger.add_scalar('{}_{}'.format(set_, meter), self.meters[set_][meter].value()[0], epoch)
                self.meters[set_][meter].reset()

        self.last_metrics.append(val_metric)
        if np.min(self.direction * np.array(self.last_metrics)) > self.direction * self.best_metric:
            self.stop_training = True

    def train_step(self, batch):
        raise NotImplementedError

    def val_step(self, batch):
        raise NotImplementedError

    def __call__(self, train_data_loader, unlabeled_train_data_loader, val_data_loader):
        self.model.to(self.device)
        iteration = 1

        for epoch in range(1, self.config['epochs']+1):
            if self.stop_training:
                break
            else:
                train_proj, train_labels = [], []
                for batch in tqdm(train_data_loader,
                                  total=len(train_data_loader),
                                  desc=f'Training epoch {epoch}'):

                    z, y, loss = self.train_step(batch)
                    train_proj.append(z)
                    train_labels.append(y)
                    self.logger.add_scalar('Loss', loss, iteration)
                    iteration += 1

                train_proj = torch.cat(train_proj).numpy()
                train_labels = torch.cat(train_labels).numpy()

                for batch in tqdm(unlabeled_train_data_loader,
                                  total=len(unlabeled_train_data_loader),
                                  desc=f'Training epoch {epoch}'):

                    loss = self.train_step(batch)
                    self.logger.add_scalar('Loss', loss, iteration)
                    iteration += 1

                val_proj, val_labels = [], []
                for batch in tqdm(val_data_loader,
                                  total=len(val_data_loader),
                                  desc=f'Validation epoch {epoch}'):
                    z, y = self.val_step(batch)
                    val_proj.append(z)
                    val_labels.append(y)

                val_proj = torch.cat(val_proj).numpy()
                val_labels = torch.cat(val_labels).numpy()
                acc_metrics = knn_pred(train_proj, train_labels, val_proj, val_labels)
                self.meters['val']['f1-score'].add(acc_metrics['avg F1'])
                self.reset_meters(epoch)

def reset_parameters(model: nn.Module, seed: int = None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.Conv1d) \
                or isinstance(m, nn.Conv2d) \
                or isinstance(m, nn.Conv3d) \
                or isinstance(m, nn.Linear) \
                or isinstance(m, nn.BatchNorm1d)\
                or isinstance(m, nn.BatchNorm2d)\
                or isinstance(m, nn.BatchNorm3d):
            m.reset_parameters()

