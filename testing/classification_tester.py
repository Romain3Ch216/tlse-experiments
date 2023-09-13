from testing.base_tester import Tester
import torch
import os


class AETester(Tester):
    def __init__(self, dataset, model, config):
        super().__init__(dataset, model, config)
        self.model = model
        self.device = config['device']
        self.test_folder = os.path.join(self.config['log_dir'], 'test_metrics')

    def test_step(self, batch):
        self.model.eval()

        x, y = batch
        x = x.to(self.device)

        with torch.no_grad():
            z, r = self.model(x)
        return z.cpu(), y.cpu()


class MAETester(Tester):
    def __init__(self, dataset, model, config):
        super().__init__(dataset, model, config)
        self.model = model
        self.device = config['device']
        self.test_folder = os.path.join(self.config['log_dir'], 'test_metrics')

    def test_step(self, batch):
        self.model.eval()

        x, y = batch
        x = x.to(self.device)

        with torch.no_grad():
            loss, pred, mask, z = self.model(x, mask_ratio=self.mask_ratio)

        if self.model.is_cls_token:
            z = z[:, 0, :]
        else:
            z = torch.mean(z[:, 1:, :], dim=1)

        return z.cpu(), y.cpu()
