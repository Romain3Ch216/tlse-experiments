import pdb
from tqdm import tqdm
from testing.base_tester import Tester
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from learning.losses import test_prototypical_loss_logits
import os
import json
import numpy as np
from utils import *


class ClassificationTester(Tester):
    def __init__(self, dataset, model, config):
        super().__init__(dataset, model, config)
        self.model = model
        self.metrics = ['cross_entropy', 'accuracy', 'f1_score']
        self.device = config['device']
        self.test_folder = os.path.join(self.config['log_dir'], 'test_metrics')
        self.logits, self.labels = [], []

    def val_step(self, batch):
        self.model.eval()
        x, y = batch
        x, y = x.to(self.device), y.to(self.device).long() - 1
        # mask = y != -1
        # x = x[mask]
        # y = y[mask]
        with torch.no_grad():
            logits = self.model(x)
        if len(logits.shape) >= 4:
            logits = logits.squeeze(1)
        if len(logits.shape) >= 3:
            logits = logits.squeeze(1)
        self.logits.append(logits.cpu())
        self.labels.append(y.view(-1).cpu())

    def compute_metrics(self):
        self.logits = torch.cat(self.logits, dim=0)
        self.labels = torch.cat(self.labels)
        pred = torch.argmax(self.logits, dim=-1)
        test_metrics = {
            'cross_entropy': F.cross_entropy(self.logits, self.labels).mean().item(),
            'accuracy': accuracy_score(pred, self.labels),
            'f1_score': f1_score(pred, self.labels, average=None)
        }
        cm = confusion_matrix(self.labels, pred)
        np.save(os.path.join(self.config['log_dir'], 'confusion_matrix.npy'), cm)
        test_metrics['f1_score'] = list(test_metrics['f1_score'])
        with open(os.path.join(self.config['log_dir'], 'test_metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=4)


class PrototypicalTester(Tester):
    def __init__(self, dataset, model, config):
        super().__init__(dataset, model, config)
        self.model = model
        self.metrics = ['cross_entropy', 'accuracy', 'f1_score']
        self.device = config['device']
        self.test_folder = os.path.join(self.config['log_dir'], 'test_metrics')
        self.logits, self.labels = [], []

    def val_step(self, batch, prototypes):
        self.model.eval()
        x, y = batch
        x, y = x.to(self.device), y.to(self.device).long().view(-1) - 1
        with torch.no_grad():
            z, _ = self.model(x)
        logits = test_prototypical_loss_logits(prototypes, z.view(z.shape[0], z.shape[-1]), y)
        self.logits.append(logits.cpu())
        self.labels.append(y.cpu())

    def compute_metrics(self):
        self.logits = torch.cat(self.logits, dim=0)
        self.labels = torch.cat(self.labels)
        pred = torch.argmax(self.logits, dim=-1)
        test_metrics = {
            'cross_entropy': F.cross_entropy(self.logits, self.labels).mean().item(),
            'accuracy': accuracy_score(pred, self.labels),
            'f1_score': f1_score(pred, self.labels, average=None)
        }
        cm = confusion_matrix(self.labels, pred)
        np.save(os.path.join(self.config['log_dir'], 'confusion_matrix.npy'), cm)
        test_metrics['f1_score'] = list(test_metrics['f1_score'])
        with open(os.path.join(self.config['log_dir'], 'test_metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=4)

    def __call__(self, test_data_loader, dataset, labels):
        self.model.to(self.device)
        prototypes = self.model.prototypes(dataset, labels)
        for batch in tqdm(test_data_loader,
                          total=len(test_data_loader),
                          desc='Testing model...'):
            self.val_step(batch, prototypes)

        self.compute_metrics()

class PatchClassificationTester(Tester):
    def __init__(self, dataset, model, config):
        super().__init__(dataset, model, config)
        self.model = model
        self.metrics = ['cross_entropy', 'accuracy', 'f1_score']
        self.device = config['device']

        self.logits = []
        self.labels = []

        self.test_folder = os.path.join(self.config['log_dir'], 'test_metrics')

    def val_step(self, batch):
        self.model.eval()
        x, y = batch
        mask = y != 0
        if mask.sum() > 0:
            x, y = x.to(self.device), y.to(self.device).long() - 1
            with torch.no_grad():
                logits = self.model(x)
            if isinstance(logits, tuple):
                logits = logits[1]
            logits = logits[mask]
            y = y[mask]
            self.logits.append(logits.cpu())
            self.labels.append(y.cpu())

    def compute_metrics(self):
        self.logits = torch.cat(self.logits, dim=0)
        self.labels = torch.cat(self.labels)
        pred = torch.argmax(self.logits, dim=-1)
        test_metrics = {
            'cross_entropy': F.cross_entropy(self.logits, self.labels).mean().item(),
            'accuracy': accuracy_score(pred, self.labels),
            'f1_score': f1_score(pred, self.labels, average=None)
        }
        test_metrics['f1_score'] = list(test_metrics['f1_score'])
        with open(os.path.join(self.config['log_dir'], 'test_metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=4)



class SemiSupervisedPatchClassificationTester(Tester):
    def __init__(self, dataset, model, config):
        super().__init__(dataset, model, config)
        self.model = model
        self.metrics = ['cross_entropy', 'mse', 'accuracy', 'f1_score']
        self.device = config['device']

        self.logits = []
        self.labels = []
        self.mse = []

        self.test_folder = os.path.join(self.config['log_dir'], 'test_metrics')

    def val_step(self, batch):
        self.model.eval()
        x, y = batch
        mask = y != 0
        if mask.sum() > 0:
            x, y = x.to(self.device), y.to(self.device).long() - 1
            with torch.no_grad():
                reconstruction, logits = self.model(x)
            logits = logits[mask]
            y = y[mask]
            self.mse.append(F.mse_loss(reconstruction, x).item())
            self.logits.append(logits.cpu())
            self.labels.append(y.cpu())

    def compute_metrics(self):
        self.logits = torch.cat(self.logits, dim=0)
        self.labels = torch.cat(self.labels)
        pred = torch.argmax(self.logits, dim=-1)
        test_metrics = {
            'cross_entropy': F.cross_entropy(self.logits, self.labels).mean().item(),
            'accuracy': accuracy_score(pred, self.labels),
            'f1_score': f1_score(pred, self.labels, average=None),
            'mse': sum(self.mse)/len(self.mse)
        }
        test_metrics['f1_score'] = list(test_metrics['f1_score'])

        with open(os.path.join(self.config['log_dir'], 'test_metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=4)

class SemiSupervisedTester(Tester):
    def __init__(self, dataset, model, config):
        super().__init__(dataset, model, config)
        self.model = model
        self.metrics = ['cross_entropy', 'accuracy', 'f1_score', 'mse']
        self.device = config['device']

        self.logits = []
        self.labels = []
        self.mse = []

        self.test_folder = os.path.join(self.config['log_dir'], 'test_metrics')

    def val_step(self, batch):
        self.model.eval()
        x, y = batch
        mask = y != 0
        x, y = x.to(self.device), y.to(self.device).long() - 1
        if mask.sum() > 0:
            with torch.no_grad():
                logits, x1, y1 = self.model(x)
            mse = F.mse_loss(x1[mask], y1[mask])
            self.mse.append(mse.item())
            self.logits.append(logits[mask].cpu())
            self.labels.append(y[mask].cpu())

    def compute_metrics(self):
        self.logits = torch.cat(self.logits)
        self.labels = torch.cat(self.labels)
        pred = torch.argmax(self.logits, dim=-1)
        test_metrics = {
            'cross_entropy': F.cross_entropy(self.logits, self.labels).mean().item(),
            'accuracy': accuracy_score(pred, self.labels),
            'f1_score': f1_score(pred, self.labels, average=None),
            'mse': sum(self.mse) / len(self.mse)
        }
        test_metrics['f1_score'] = list(test_metrics['f1_score'])

        with open(os.path.join(self.config['log_dir'], 'test_metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=4)


class SemiSupervisedVAETester(Tester):
    def __init__(self, dataset, model, config):
        super().__init__(dataset, model, config)
        self.model = model
        self.metrics = ['cross_entropy', 'accuracy', 'f1_score', 'mse', 'sam']
        self.device = config['device']

        self.logits = []
        self.labels = []
        self.mse = []
        self.sam = []
        self.y_dim = dataset.n_classes

        self.test_folder = os.path.join(self.config['log_dir'], 'test_metrics')

    def val_step(self, batch):
        self.model.eval()
        x, y = batch
        x = x.squeeze(1)
        x, y = x.to(self.device), y.to(self.device).long() - 1

        y = one_hot(y, self.y_dim).to(self.device)
        with torch.no_grad():
            reconstruction_l = self.model(x, y)

        mse_l = F.mse_loss(x, reconstruction_l)
        sam_l = sam_(x, reconstruction_l)

        with torch.no_grad():
            logits_l = self.model.q_y_x_batch(x)
        Lc_l = F.cross_entropy(logits_l, torch.argmax(y, dim=-1))

        self.mse.append(mse_l.item())
        self.sam.append(sam_l.item())
        self.logits.append(logits_l.cpu())
        self.labels.append(torch.argmax(y, dim=-1).cpu())

    def compute_metrics(self):
        self.logits = torch.cat(self.logits)
        self.labels = torch.cat(self.labels)
        pred = torch.argmax(self.logits, dim=-1)
        test_metrics = {
            'cross_entropy': F.cross_entropy(self.logits, self.labels).mean().item(),
            'accuracy': accuracy_score(pred, self.labels),
            'f1_score': f1_score(pred, self.labels, average=None),
            'mse': sum(self.mse) / len(self.mse),
            'sam': sum(self.sam) / len(self.sam)
        }
        test_metrics['f1_score'] = list(test_metrics['f1_score'])
        cm = confusion_matrix(self.labels, pred)
        np.save(os.path.join(self.config['log_dir'], 'confusion_matrix.npy'), cm)

        with open(os.path.join(self.config['log_dir'], 'test_metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=4)


class SiameseTester(Tester):
    def __init__(self, dataset, model, config):
        super().__init__(dataset, model, config)
        self.model = model
        self.metrics = ['cross_entropy', 'accuracy', 'f1_score']
        self.device = config['device']

        self.logits = []
        self.labels = []

        self.test_folder = os.path.join(self.config['log_dir'], 'test_metrics')

    def val_step(self, batch):
        self.model.eval()
        x, labels = batch
        x = x.to(self.device)
        labels = (labels-1).to(self.device).long()
        mask = labels != -1
        with torch.no_grad():
            y, r_logits, logits, reconstruction = self.model(x)
        logits = logits[mask]
        labels = labels[mask]
        self.logits.append(logits.cpu())
        self.labels.append(labels.cpu())

    def compute_metrics(self):
        self.logits = torch.cat(self.logits, dim=0)
        self.labels = torch.cat(self.labels)
        pred = torch.argmax(self.logits, dim=-1)
        test_metrics = {
            'cross_entropy': F.cross_entropy(self.logits, self.labels).mean().item(),
            'accuracy': accuracy_score(pred, self.labels),
            'f1_score': f1_score(pred, self.labels, average=None)
        }
        test_metrics['f1_score'] = list(test_metrics['f1_score'])
        with open(os.path.join(self.config['log_dir'], 'test_metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=4)

