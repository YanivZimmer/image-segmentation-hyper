import torch
from tqdm import tqdm

from untils.average_meter import AverageMeter
#from Losses import ComboLoss, dice_metric
from untils.loss import dice_coefficient


def acc_metric(input, target):
    inp = torch.where(input>0.5, torch.tensor(1, device='cuda'), torch.tensor(0, device='cuda'))
    acc = (inp.squeeze(1) == target).float().mean()
    return acc
# from https://www.kaggle.com/rishabhiitbhu/unet-with-resnet34-encoder-pytorch
def my_acc(input, target):
    ignore_label = 0
    mask = (target != ignore_label).view(-1)
    input = torch.argmax(input.permute(0,2,3,1), dim=-1)#.permute(0,2,3,1)
    res = sum(input.view(-1)[mask] == target.view(-1)[mask])
    return res/sum(mask)
    # not_ignored = (target != ignore_label)
    # equal_indices = torch.eq(target[not_ignored], input[not_ignored])
    # count_equal_indices = torch.sum(equal_indices).item()
    # return count_equal_indices/not_ignored.shape[0]

def metric(probability, truth, threshold=0.5, reduction='none'):
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert(probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice
def dice_ignore_label(pred, target, ignore_label=0):
    smooth = 0.0

    # Create a mask to ignore pixels with the specified label
    ignore_mask = (target != ignore_label).float()
    x = torch.argmax(pred, dim=1).view(-1)
    # Calculate intersection, union, and dice
    intersection = torch.sum((x == target.view(-1)) * ignore_mask)
    union = torch.sum((x + target.view(-1)) * ignore_mask)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

def evaluate(valid_loader, model, device='cuda', metric=None):#=dice_metric):
    losses = AverageMeter()
    model = model.to(device)
    model.eval()
    tk0 = tqdm(valid_loader, total=len(valid_loader))
    with torch.no_grad():
        for b_idx, data in enumerate(tk0):
            for key, value in data.items():
                data[key] = value.to(device)
            out = model(data['image'])
            #my_dice = dice_coefficient(out, data['mask'])
            acc = my_acc(out, data['mask'])
            losses.update(acc, valid_loader.batch_size)
            #out   = torch.sigmoid(out)
            #dice  = metric(out, data['mask']).cpu()
            #losses.update(dice.mean().item(), valid_loader.batch_size)
            tk0.set_postfix(dice_score=losses.avg)
    return losses.avg