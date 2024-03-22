from learning.autoencoding_trainer import AutoEncodingTrainer, MAETrainer
import numpy as np


def load_trainer(dataset, model, config, E_dir=None, E_dif=None, theta=None, data_augmentation=None):
    if config['model'] == 'MAE':
        trainer = MAETrainer(dataset, model, config)
    elif config['model'] == 'AE':
        trainer = AutoEncodingTrainer(dataset, model, config)
    return trainer
