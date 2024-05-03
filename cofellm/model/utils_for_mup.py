import math
import torch
import warnings
from torch.optim.lr_scheduler import _LRScheduler

def _exact(p, mean, std):
    # Experimental improvements: make final mean/variance exactly equal to the target.
    # Sometimes large variance in normal_() causes the actual mean to deviate from expected.
    p.data = (p.data - torch.mean(p, dtype=torch.float32).item()) / math.sqrt(torch.var(p).item()) * std + mean


class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)
        assert "width_mult" in self.optimizer.param_groups[0].keys()

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        elif self._step_count == 1 and self.last_epoch > 0:
            return [self.eta_min / group["width_mult"] + (base_lr - self.eta_min / group["width_mult"]) *
                    (1 + math.cos((self.last_epoch) * math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min / group["width_mult"]) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (group['lr'] - self.eta_min / group["width_mult"]) + self.eta_min / group["width_mult"]
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [self.eta_min / group["width_mult"] + (base_lr - self.eta_min / group["width_mult"]) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]