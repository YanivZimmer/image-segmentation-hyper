import torch
import torch.nn as nn
from collections import defaultdict
import torchvision
import torch.nn.functional as F
import torch
from torchmetrics import Dice
from untils.metrics import MulticlassDiceLoss
from untils.metrics import IouCalculator

IGNORE_LABEL = 0
NUM_CLASSES=11

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
        input = torch.argmax(input,axis=1)
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

import torch
import torch.nn as nn
import torch.nn.functional as F

class IoULoss(nn.Module):
    def __init__(self, ignore_index=0):
        super(IoULoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        # Apply softmax to get probabilities
        input = F.softmax(input, dim=1)

        # Flatten the tensors
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, input.shape[1])
        target = target.view(-1)
       
        # Create a mask to ignore the specified label
        mask = (target != self.ignore_index)
       
        # Apply the mask
        input = input[mask]
        target = target[mask]
       
        # Convert target to one-hot encoding
        target_one_hot = F.one_hot(target, num_classes=input.shape[1]).float()
       
        # Compute the intersection and union
        intersection = (input * target_one_hot).sum(dim=0)
        union = input.sum(dim=0) + target_one_hot.sum(dim=0) - intersection
       
        # Compute the IoU and return the loss
        iou = (intersection + 1e-6) / (union + 1e-6)
        #iou_loss = 1 - iou.mean()
        iou_loss = - iou.mean().log()
        return iou_loss


class DiceLoss(nn.Module):
    def __init__(self, ignore_index=0, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, input, target):
        # Apply softmax to get probabilities
        input = F.softmax(input, dim=1)

        # Flatten the tensors
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, input.shape[1])
        target = target.view(-1)
       
        # Create a mask to ignore the specified label
        mask = (target != self.ignore_index)
       
        # Apply the mask
        input = input[mask]
        target = target[mask]
       
        # Convert target to one-hot encoding
        target_one_hot = F.one_hot(target, num_classes=input.shape[1]).float()
       
        # Compute the intersection and the union
        intersection = (input * target_one_hot).sum(dim=0)
        union = input.sum(dim=0) + target_one_hot.sum(dim=0)
       
        # Compute the Dice coefficient and return the loss
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = - dice.mean().log()
        return dice_loss

class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma, device="cuda"):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=0)#nn.BCEWithLogitsLoss(ignore_index=0)#
        self.dice_score = Dice(average="micro", ignore_index=0).to(device)
        self.iou_calculator = IouCalculator()
        self.dicer = MulticlassDiceLoss( NUM_CLASSES, softmax_dim=1)
        self.ioul=IoULoss()
        self.dicel=DiceLoss()
        # Calculate the loss

    def dice_loss(self, input, target):
        return 1 - self.dice_score(input, target)

    def forward(self, input, target):
        #import pdb; pdb.set_trace()
        # loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        # return loss.mean()
        # return dice_loss(input, target)
        #return - torch.log(self.iou_calculator.calculate_iou(input, target))
        loss= self.dicel(input, target)
        #loss= self.ioul(input, target)
        #loss = self.cross_entropy(input, target)#-torch.log(self.dice_score(input, target))
        return loss
