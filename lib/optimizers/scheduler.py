import torch
from collections import Counter

class MultiStepLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, reset_threshold=0):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.reset_threshold=reset_threshold
        super(MultiStepLR, self).__init__(optimizer, last_epoch)

    def get_last_epoch(self):
        return self.last_epoch - self.reset_threshold

    def get_lr(self):
        if self.get_last_epoch() not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.get_last_epoch()]
                for group in self.optimizer.param_groups]