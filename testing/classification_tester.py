import pdb
from tqdm import tqdm
from testing.base_tester import Tester
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
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


