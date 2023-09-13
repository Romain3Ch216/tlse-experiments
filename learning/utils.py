from learning.autoencoding_trainer import AutoEncodingTrainer, MAETrainer
import numpy as np


def load_trainer(dataset, model, config, E_dir=None, E_dif=None, theta=None, data_augmentation=None):
    if config['model'] == 'mae':
        trainer = MAETrainer(dataset, model, config)
    elif config['model'] == 'AE':
        trainer = AutoEncodingTrainer(dataset, model, config)
    return trainer


def compute_imf_weights(n_samples, n_classes=None, ignored_classes=[]):
    """ Compute inverse median frequency weights for class balancing.
    """
    weights = np.zeros_like(n_samples)
    # Normalize the pixel counts to obtain frequencies
    frequencies = n_samples / np.sum(n_samples)
    # Obtain the median on non-zero frequencies
    idx = np.nonzero(frequencies)
    median = np.median(frequencies[idx])
    weights[idx] = median / frequencies[idx]
    weights[frequencies == 0] = 0.
    weights = [float(w) for w in weights]
    return weights
