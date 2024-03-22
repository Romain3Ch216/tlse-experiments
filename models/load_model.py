from models.transformer import MaskedAutoencoder
from models.autoencoder import AutoEncoder
import torch


def load_model(config):
    if config['model'] == 'MAE':
        if config['seed'] is not None:
            torch.manual_seed(config['seed'])
            torch.cuda.manual_seed(config['seed'])

        model = MaskedAutoencoder(
            n_bands=config['n_bands'],
            seq_size=config['seq_size'],
            in_chans=1,
            embed_dim=config['embed_dim'],
            depth=config['depth'],
            num_heads=config['n_heads'],
            decoder_embed_dim=config['decoder_embed_dim'],
            decoder_depth=config['decoder_depth'],
            decoder_num_heads=config['decoder_n_heads'],
            mlp_ratio=config['mlp_ratio'],
            cls_token=config['cls_token']
            )
    elif config['model'] == 'AE':
        model = AutoEncoder(
            n_channels=config['n_bands'],
            z_dim=config['z_dim'],
            h_dim=config['h_dim'],
            decoder_h_dim=config['decoder_h_dim']
        )

    return model
