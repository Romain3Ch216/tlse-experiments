import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import json
import os
import collections


class Tester(object):
    def __init__(self, dataset, model, config):
        self.metrics = None
        self.device = None
        self.model = model
        self.dataset = dataset
        self.config = config
        self.model.eval()

    def val_step(self, batch):
        raise NotImplementedError

    def compute_metrics(self):
        raise NotImplementedError

    def __call__(self, test_data_loader, dataset, labels):
        self.model.to(self.device)
        for batch in tqdm(test_data_loader,
                          total=len(test_data_loader),
                          desc='Testing model...'):
            self.val_step(batch)

        self.compute_metrics()
