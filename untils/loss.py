import torch
import torch.nn as nn
from collections import defaultdict
import torchvision
import torch.nn.functional as F
import torch
from torchmetrics import Dice

IGNORE_LABEL = 0


def dice_loss_old(input, target):
    # input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


def dice_loss(input, target, ignore_label=0):
    # Calculate Dice loss
    dice_loss = 1.0 - dice_coefficient(input, target, ignore_label)
    return dice_loss


def dice_coefficient(preds, target, ignore_label=0):
    dice = Dice(average="micro", ignore_label=ignore_label)
    res = dice(preds, target)
    return res


def dice_coefficient_old(input, target, ignore_label=0):
    smooth = 1.0
    dims = input.size()
    # Calculate the product of all dimensions except the last one
    w = torch.prod(torch.tensor(dims[:-1]))
    iflat = input.view((w, dims[-1]))
    tflat = target.view((w, dims[-1]))
    # Create only-labeled mask
    temp_mask = (torch.argmax(target, dim=-1).view(-1) != ignore_label).float()
    ignore_mask = torch.zeros((temp_mask.shape[0], dims[-1]), device=iflat.device)
    ignore_mask[temp_mask == 1, :] = 1
    # Ignore pixels with target label 0
    iflat = iflat * ignore_mask
    tflat = tflat * ignore_mask

    intersection = (iflat * tflat).sum()
    # Calculate Dice coefficient
    dice_coefficient = (2.0 * intersection + smooth) / (
        iflat.sum() + tflat.sum() + smooth
    )
    return dice_coefficient


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError(
                "Target size ({}) must be the same as input size ({})".format(
                    target.size(), input.size()
                )
            )
        max_val = (-input).clamp(min=0)
        loss = (
            input
            - input * target
            + max_val
            + ((-max_val).exp() + (-input - max_val).exp()).log()
        )
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma, device="cuda"):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=0)
        self.dice_score = Dice(average="micro", ignore_index=0).to(device)
        # Calculate the loss

    def dice_loss(self, input, target):
        return 1 - self.dice_score(input, target)

    def forward(self, input, target):
        # loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        # return loss.mean()
        # return dice_loss(input, target)
        loss = self.cross_entropy(input, target) + self.dice_loss(input, target)
        return loss
