import os
from pathlib import Path

import numpy as np
import torch.nn as nn
from collections import defaultdict
import torchvision
import torch.nn.functional as F

import torch
from tqdm import tqdm

from untils.average_meter import AverageMeter


class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.0001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                "EarlyStopping counter: {} out of {}".format(
                    self.counter, self.patience
                )
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        model_path = Path(model_path)
        parent = model_path.parent
        os.makedirs(parent, exist_ok=True)
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print(
                "Validation score improved ({} --> {}). Model saved at at {}!".format(
                    self.val_score, epoch_score, model_path
                )
            )
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score


def train_one_epoch(
    train_loader, model, optimizer, averager, loss_fn,lam = 0.1, accumulation_steps=1, device="cuda"
):
     #0.5, 0.25, 0.35
    losses = AverageMeter()
    model = model.to(device)
    model.train()
    if accumulation_steps > 1:
        optimizer.zero_grad()
    tk0 = tqdm(train_loader, total=len(train_loader))
    for b_idx, data in enumerate(tk0):
        for key, value in data.items():
            data[key] = value.to(device)
        if accumulation_steps == 1 and b_idx == 0:
            optimizer.zero_grad()
        out = model(data["image"])
        loss = loss_fn(out, data["mask"])
        if model.band_selection:
            regu = lam * model.ehbs.regularizer()
            loss += regu
        #with torch.set_grad_enabled(True):
        if True:
          loss.backward()
          if (b_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            if averager is not None:
              averager.step()
            optimizer.zero_grad()
        losses.update(loss.item(), train_loader.batch_size)
        tk0.set_postfix(loss=losses.avg, learning_rate=optimizer.param_groups[0]["lr"])
    return losses.avg
