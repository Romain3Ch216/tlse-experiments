import os
import json
from tqdm import tqdm
import torch
from learning.knn import knn_pred


class Tester(object):
    def __init__(self, dataset, model, config):
        self.metrics = None
        self.device = None
        self.model = model
        self.dataset = dataset
        self.config = config
        self.model.eval()

    def test_step(self, batch):
        raise NotImplementedError

    def __call__(self, train_data_loader, test_data_loader):
        self.model.to(self.device)

        train_proj, train_labels = [], []
        for batch in tqdm(train_data_loader,
                          total=len(train_data_loader)):

            z, y = self.test_step(batch)
            train_proj.append(z)
            train_labels.append(y)

        train_proj = torch.cat(train_proj).numpy()
        train_labels = torch.cat(train_labels).numpy()

        test_proj, test_labels = [], []
        for batch in tqdm(test_data_loader,
                          total=len(test_data_loader)):
            z, y = self.test_step(batch)
            test_proj.append(z)
            test_labels.append(y)

        test_proj = torch.cat(test_proj).numpy()
        test_labels = torch.cat(test_labels).numpy()

        acc_metrics = knn_pred(train_proj, train_labels, test_proj, test_labels)

        with open(os.path.join(self.config['log_dir'], 'test_metrics.json'), 'w') as f:
            json.dump(acc_metrics, f, indent=4)
