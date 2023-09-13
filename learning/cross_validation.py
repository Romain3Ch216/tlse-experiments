import optuna
from optuna.samplers import RandomSampler, TPESampler, GridSampler
from learning.utils import load_trainer
from models.load_model import load_model
import datetime
import numpy as np
import torch
import os


class CrossValidationObjective:
    def __init__(self, config, params_space, dataset, labeled_loader, unlabeled_loader, val_loader):
        self.config = config
        self.params_space = params_space
        self.dataset = dataset
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.val_loader = val_loader
        self.epochs = config['cv_epochs']

    def __call__(self, trial):
        config = dict((k, v) for (k, v) in self.config.items())
        for param in self.params_space:
            if self.params_space[param][0] == 'float':
                config.update({
                    param: trial.suggest_float(param, self.params_space[param][1][0], self.params_space[param][1][1])
                })
            elif self.params_space[param][0] == 'log':
                config.update({
                    param: trial.suggest_loguniform(param, self.params_space[param][1][0], self.params_space[param][1][1])
                })
            elif self.params_space[param][0] == 'int':
                if config['model'] == 'MAE':
                    if self.params_space['embed_dim'][1] < self.params_space['decoder_embed_dim'][1]:
                        self.params_space['embed_dim'][1], self.params_space['decoder_embed_dim'][1] = \
                            self.params_space['decoder_embed_dim'][1], self.params_space['embed_dim'][1]
                elif config['model'] == 'AE':
                    if self.params_space['h_dim'][1] < self.params_space['decoder_h_dim'][1]:
                        self.params_space['h_dim'][1], self.params_space['decoder_h_dim'][1] = \
                            self.params_space['decoder_h_dim'][1], self.params_space['h_dim'][1]

                config.update({
                    param: trial.suggest_categorical(param, self.params_space[param][1])
                })

        seed = np.random.randint(0, 1e8)
        trial.set_user_attr('seed', seed)
        config.update({'seed': seed})

        timestamp = datetime.datetime.now().strftime("%H%M%S")
        config['log_dir'] = os.path.join(self.config['log_dir'], 'trial_{}'.format(timestamp))
        config['save_best_model'] = False
        config['epochs'] = self.epochs
        model = load_model(config)
        trainer = load_trainer(self.dataset, model, config)
        trainer(self.labeled_loader, self.unlabeled_loader, self.val_loader)
        return trainer.best_metric


class CrossValidation:
    def __init__(self, direction='minimize', sampler='random', pruner='median'):
        self.direction = direction
        self.sampler = sampler

        if pruner == 'median':
            self.pruner = optuna.pruners.MedianPruner()
        else:
            raise NotImplementedError

    def __call__(self, config, params_space, dataset, labeled_loader, unlabeled_loader, val_loader):
        if self.sampler == 'random':
            sampler = RandomSampler()
        elif self.sampler == 'grid_search':
            grid_space = dict((k, v[1]) for (k, v) in params_space.items())
            sampler = GridSampler(grid_space)
        elif self.sampler == 'tpe':
            sampler = TPESampler()
        else:
            raise NotImplementedError
        study = optuna.create_study(direction=self.direction, sampler=sampler, pruner=self.pruner)
        study.optimize(CrossValidationObjective(config, params_space, dataset, labeled_loader, unlabeled_loader, val_loader),
                       n_trials=config['n_trials'])
        return study
