import torch.nn as nn
import torch
import numpy as np
from typing import List


def reset_parameters(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Conv1d) \
                or isinstance(m, nn.Conv2d) \
                or isinstance(m, nn.Conv3d) \
                or isinstance(m, nn.Linear) \
                or isinstance(m, nn.BatchNorm1d)\
                or isinstance(m, nn.BatchNorm2d)\
                or isinstance(m, nn.BatchNorm3d):
            m.reset_parameters()


class SpectralWrapper(nn.Module):
    """
    Converts a dict of CNNs (one for each continous spectral domain)
    into a single CNN.
    """
    def __init__(self, models, out_channels=False):
        super(SpectralWrapper, self).__init__()
        self.models = nn.ModuleDict(models)
        self.out_channels = out_channels

    @property
    def out_channels(self):
        with torch.no_grad():
            n_channels = sum([model.n_channels for model in self.models.values()])
            x = torch.ones((2, n_channels))
            x = self.forward(x)
        return x.numel()//2

    def forward(self, x):
        z, B = {}, 0

        for model_id, model in self.models.items():
            z[model_id] = model(x[:, :, B:B+model.n_channels])
            B += model.n_channels

        keys = list(z.keys())
        out_channels = [z[keys[i]].shape[-1] for i in range(len(z))]
        out = torch.cat([z[keys[i]] for i in range(len(z))], dim=-1)
        if self.out_channels:
            return out, out_channels
        else:
            return out


def get_continuous_bands(bbl: np.ndarray) -> List[int]:
    n_bands = []
    good_bands = np.where(bbl == True)[0]
    s = 1
    for i in range(len(good_bands)-1):
        if good_bands[i] == good_bands[i+1]-1:
            s += 1
        else:
            n_bands.append(s)
            s = 1

    n_bands.append(s)
    return n_bands


class CnnWrapper(nn.Module):
    """
    Converts a dict of CNNs (one for each continous spectral domain)
    into a single CNN.
    """
    def __init__(self, models, flatten=True, conv_dropout=False):
        super(CnnWrapper, self).__init__()
        self.models = nn.ModuleDict(models)
        self.flatten = flatten
        self.conv_dropout = conv_dropout

    @property
    def out_channels(self):
        with torch.no_grad():
            n_channels = sum([model.n_channels for model in self.models.values()])
            x = torch.ones((2, n_channels))
            x = self.forward(x)
        return x.numel()//2

    def forward(self, x):
        if len(x.shape)>2:
            patch_size = x.shape[-1]
            x = x[:,0,:,patch_size//2, patch_size//2]
        z, B = {}, 0

        for model_id, model in self.models.items():
            z[model_id] = model(x[:, B:B+model.n_channels])
            B += model.n_channels

        keys = list(z.keys())
        if self.conv_dropout:
            dropout = torch.ones(len(keys))
            dropout[np.random.randint(len(z))] = 0
            out = torch.cat([z[keys[i]]*dropout[i] for i in range(len(z))], dim=-1)
        else:
            out = torch.cat([z[keys[i]] for i in range(len(z))], dim=-1)

        if self.flatten:
            out = out.view(out.shape[0], -1)
        return out


class View(torch.nn.Module):
    def forward(self, x):
        return x.squeeze(-1).squeeze(-1)

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, _, _, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, None, None, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return self.cached_penc
