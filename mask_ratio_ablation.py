import torch
from TlseHypDataSet.tlse_hyp_data_set import TlseHypDataSet
from TlseHypDataSet.utils.dataset import DisjointDataSplit
from learning.cross_validation import CrossValidation
from learning.utils import load_trainer
from testing.utils import load_tester
from utils import load_config
import datetime
import os
import argparse
from utils import update_config
from models.load_model import load_model
import numpy as np


def main(config):
    config['model'] = 'MAE'
    config = load_config(config)

    dataset = TlseHypDataSet(
        config['dataset_path'],
        pred_mode=config['pred_mode'],
        patch_size=config['patch_size'],
        in_h5py=config['h5py'],
        data_on_gpu=config['data_on_gpu'],
    )

    print(f'Dataset has {len(dataset)} samples.')

    base_log_dir = os.path.join(config['root_path'], config['model'], datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # EXP 1
    config.update({
        'n_bands': dataset.n_bands,
        'n_classes': dataset.n_classes,
        'E_dir': dataset.E_dir,
        'E_dif': dataset.E_dif,
        'theta': dataset.theta,
        'bbl': dataset.bbl,
        'lr': 1e-4,
        'weight_decay': 1e-10,
        'embed_dim': 32,
        'n_heads': 4,
        'decoder_embed_dim': 16,
        'seq_size': 5,
        'depth': 4,
        'decoder_depth': 3,
        'decoder_n_heads': 4,
        'mlp_ratio': 4,
        'cls_token': True,
        'save_best_model': True,
        'reduce_cls': False,
        'z_dim': None
    })

    splits = [DisjointDataSplit(dataset, split=default_split) for default_split in dataset.standard_splits]
    n_runs = 5

    for run in range(n_runs):
        seed = np.random.randint(1e6)
        for mask_ratio in [0.7, 0.8, 0.9]:
            for split_id, split in enumerate(splits):
                config.update({
                    'log_dir': os.path.join(base_log_dir, f'split_{split_id+1}_mask_ratio_{mask_ratio}_seed_{seed}'),
                    'mask_ratio': mask_ratio,
                    'seed': seed
                })

                labeled_loader = torch.utils.data.DataLoader(split.sets_['train'], shuffle=True, batch_size=config['batch_size'], pin_memory=True)
                unlabeled_data = torch.utils.data.ConcatDataset([split.sets_['labeled_pool'], split.sets_['unlabeled_pool']])
                unlabeled_loader = torch.utils.data.DataLoader(unlabeled_data, shuffle=True, batch_size=config['batch_size'], pin_memory=True)
                val_loader = torch.utils.data.DataLoader(split.sets_['validation'], shuffle=False, batch_size=config['batch_size'], pin_memory=True)
                test_loader = torch.utils.data.DataLoader(split.sets_['test'], shuffle=False, batch_size=config['batch_size'], pin_memory=True)

                model = load_model(config)
                trainer = load_trainer(dataset, model, config)
                trainer(labeled_loader, unlabeled_loader, val_loader)

                # Test the model on the test set
                best_params = torch.load(os.path.join(config['log_dir'], 'best_model.pth.tar'), map_location=config['device'])
                trainer.model.load_state_dict(best_params['state_dict'])
                tester = load_tester(dataset, trainer.model, config)
                tester(labeled_loader, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='Path to root')
    parser.add_argument('--device', type=str, default='cpu', help="Specify cpu or gpu")

    # Model options
    model_options = parser.add_argument_group('Model')
    model_options.add_argument('--restore', type=str)

    # Training options
    training_options = parser.add_argument_group('Training')
    training_options.add_argument('--epochs', type=int, default=100)
    training_options.add_argument('--cv_epochs', type=int, default=15)
    training_options.add_argument('--lr', type=float, default=1e-4)
    training_options.add_argument('--batch_size', type=int, default=256)
    training_options.add_argument('--n_trials', type=int, default=10)
    training_options.add_argument('--patience', type=int, default=10)
    training_options.add_argument('--seed', type=int, default=None)
    training_options.add_argument('--max_batch', type=str, default=None)

    # Data set options
    data_options = parser.add_argument_group('Dataset')
    data_options.add_argument('--h5py', action='store_true')
    data_options.add_argument('--data_on_gpu', action='store_true')

    args = parser.parse_args()
    config = parser.parse_args()
    config = vars(config)

    config.update({
        'root_path': os.path.join(config['root'], 'experiments'),
        'dataset_path': os.path.join(config['root'], 'datasets/Toulouse'),
    })

    main(config)
