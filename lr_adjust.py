
import numpy as np
import torch


class EarlyAjust:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta
        self.earlyadj = 0

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.delta:
            self.counter += 1
            print('EarlyAdjust counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.earlyadj += 1
                self.counter = 0
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0


e = EarlyAjust()
a = [1.14, 1.13, 1.12, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.0, 0.9]
for i, aa in enumerate(a):
    print(i, aa)
    e(aa)

