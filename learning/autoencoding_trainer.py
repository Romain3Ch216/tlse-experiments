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
        x = x.to(self.device)

        z, r = self.model(x)
        loss = F.mse_loss(x, r)
        loss.backward()
        self.optimizer.step()
        return z.detach().cpu(), y.cpu(), loss.item()

    def val_step(self, batch):
        self.model.eval()

        x, y = batch
        x = x.to(self.device)

        with torch.no_grad():
            z, r = self.model(x)
        loss = self.criterion(x, r)
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
        x = x.to(self.device)

        loss, pred, mask, z = self.model(x, mask_ratio=self.mask_ratio)

        if self.model.is_cls_token:
            z = z[:, 0, :]
        else:
            z = torch.mean(z[:, 1:, :], dim=1)

        loss.backward()
        self.optimizer.step()
        return z.detach().cpu(), y.cpu(), loss.item()

    def val_step(self, batch):
        self.model.eval()

        x, y = batch
        x = x.to(self.device)

        with torch.no_grad():
            loss, pred, mask, z = self.model(x, mask_ratio=self.mask_ratio)

        if self.model.is_cls_token:
            z = z[:, 0, :]
        else:
            z = torch.mean(z[:, 1:, :], dim=1)

        self.meters['val']['reconstruction'].add(loss.item())
        return z.cpu(), y.cpu()
