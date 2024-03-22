import torch
import torch.nn.functional as F
import torchnet
from learning.base_trainer import Trainer


class AutoEncodingTrainer(Trainer):
    def __init__(self, dataset, model, config):
        super().__init__(dataset, model, config)
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        self.metrics = ['reconstruction', 'f1-score']
        self.meters = {
            'train': {},
            'val': dict((metric, torchnet.meter.AverageValueMeter()) for metric in self.metrics)
        }
        self.device = config['device']
        self.target_metric = 'f1-score'
        self.best_metric = 0
        self.direction = -1

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        x, y = batch
        x = x.to(self.device).view(x.shape[0], x.shape[-1])

        z, r = self.model(x)
        loss = F.mse_loss(x, r)
        loss.backward()
        self.optimizer.step()
        return z.detach().cpu(), y.cpu(), loss.item()

    def val_step(self, batch):
        self.model.eval()

        x, y = batch
        x = x.to(self.device).view(x.shape[0], x.shape[-1])

        with torch.no_grad():
            z, r = self.model(x)
        loss = F.mse_loss(x, r)
        self.meters['val']['reconstruction'].add(loss.item())
        return z.cpu(), y.cpu()


class MAETrainer(Trainer):
    def __init__(self, dataset, model, config):
        super().__init__(dataset, model, config)
        self.model = model
        self.mask_ratio = config['mask_ratio']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

        self.metrics = ['reconstruction', 'f1-score']
        self.meters = {
            'train': {},
            'val': dict((metric, torchnet.meter.AverageValueMeter()) for metric in self.metrics)
        }
        self.device = config['device']
        self.target_metric = 'f1-score'
        self.best_metric = 0
        self.direction = -1


    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        x, y = batch
        x = x.to(self.device).view(x.shape[0], x.shape[-1])

        loss, _, _, z = self.model(x, mask_ratio=self.mask_ratio)
        loss.backward()
        self.optimizer.step()
        return z.detach().cpu(), y.cpu(), loss.item()

    def val_step(self, batch):
        self.model.eval()

        x, y = batch
        x = x.to(self.device).view(x.shape[0], x.shape[-1])

        with torch.no_grad():
            loss, _, _, z = self.model(x, mask_ratio=self.mask_ratio)

        self.meters['val']['reconstruction'].add(loss.item())
        return z.cpu(), y.cpu()
