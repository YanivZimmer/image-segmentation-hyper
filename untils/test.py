import torch
from tqdm import tqdm
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassPrecision, MulticlassRecall,MulticlassAccuracy
from contextlib import redirect_stdout
from untils.average_meter import AverageMeter

# from Losses import ComboLoss, dice_metric
from untils.loss import dice_coefficient
from untils.metrics import IouCalculator


def acc_metric(input, target):
    inp = torch.where(
        input > 0.5, torch.tensor(1, device="cuda"), torch.tensor(0, device="cuda")
    )
    acc = (inp.squeeze(1) == target).float().mean()
    return acc


# from https://www.kaggle.com/rishabhiitbhu/unet-with-resnet34-encoder-pytorch
def my_acc(input, target):
    ignore_label = 0
    mask = (target != ignore_label).view(-1)
    input = torch.argmax(input.permute(0, 2, 3, 1), dim=-1)  # .permute(0,2,3,1)
    res = sum(input.view(-1)[mask] == target.view(-1)[mask])
    return res / sum(mask)
    # not_ignored = (target != ignore_label)
    # equal_indices = torch.eq(target[not_ignored], input[not_ignored])
    # count_equal_indices = torch.sum(equal_indices).item()
    # return count_equal_indices/not_ignored.shape[0]


def metric(probability, truth, threshold=0.5, reduction="none"):
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert probability.shape == truth.shape

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

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


def evaluate_old(valid_loader, model, device="cuda", metric=None):  # =dice_metric):
    iou_calculator = IouCalculator()
    losses = AverageMeter()
    IoU = AverageMeter()
    dice = AverageMeter()
    model = model.to(device)
    model.eval()
    tk0 = tqdm(valid_loader, total=len(valid_loader))
    with torch.no_grad():
        for b_idx, data in enumerate(tk0):
            for key, value in data.items():
                data[key] = value.to(device)
            out = model(data["image"])
            # my_dice = dice_coefficient(out, data['mask'])
            acc = my_acc(out, data["mask"])
            iou = iou_calculator.calculate_iou(out, data["mask"])
            IoU.update(iou, valid_loader.batch_size)
            losses.update(acc, valid_loader.batch_size)
            # out   = torch.sigmoid(out)
            # dice  = metric(out, data['mask']).cpu()
            # losses.update(dice.mean().item(), valid_loader.batch_size)
            tk0.set_postfix(acc_score=losses.avg, iou_score=IoU.avg)
    return losses.avg, IoU.avg

def evaluate(valid_loader, model,num_classes, output_file=None, device="cuda"):
    model.eval()
    tk0 = tqdm(valid_loader, total=len(valid_loader))
    jaccard = MulticlassJaccardIndex(num_classes=num_classes, average='micro', ignore_index=0).to(device)
    precision = MulticlassPrecision(num_classes=num_classes, average='micro', ignore_index=0).to(device)
    recall = MulticlassRecall(num_classes=num_classes, average='micro', ignore_index=0).to(device)
    accuracy = MulticlassAccuracy(num_classes=num_classes, average='micro', ignore_index=0).to(device)

    jaccard_mean = MulticlassJaccardIndex(num_classes=num_classes,average='macro', ignore_index=0).to(device)
    precision_mean = MulticlassPrecision(num_classes=num_classes, average='macro', ignore_index=0).to(device)
    recall_mean = MulticlassRecall(num_classes=num_classes, average='macro', ignore_index=0).to(device)
    accuracy_mean = MulticlassAccuracy(num_classes=num_classes, average='macro', ignore_index=0).to(device)

    jaccard_weighted = MulticlassJaccardIndex(num_classes=num_classes, average='weighted', ignore_index=0).to(device)
    precision_weighted = MulticlassPrecision(num_classes=num_classes, average='weighted', ignore_index=0).to(device)
    recall_weighted = MulticlassRecall(num_classes=num_classes, average='weighted', ignore_index=0).to(device)
    accuracy_weighted = MulticlassAccuracy(num_classes=num_classes, average='weighted', ignore_index=0).to(device)

    # Accumulate counts of pixels for each class across all batches
    total_pixels_per_class = torch.zeros(num_classes, device=device)
    total_num_pixels = 0

    with torch.no_grad():
        for b_idx, data in enumerate(tk0):
            for key, value in data.items():
                data[key] = value.to(device)
            out = model(data["image"])
            # Mask out pixels with label 0
            mask = data["mask"]
            valid_mask = mask != 0
            #print("valid_mask1",valid_mask[0])
            # Flatten tensors
            out = out.argmax(dim=1).flatten()  # Get the predicted classes
            mask = mask.flatten()
            valid_mask = valid_mask.flatten()
            # Apply the valid mask
            valid_out = out[valid_mask]
            valid_mask = mask[valid_mask]

            # Update metrics for the current batch
            jaccard.update(valid_out, valid_mask)
            precision.update(valid_out, valid_mask)
            recall.update(valid_out, valid_mask)
            accuracy.update(valid_out, valid_mask)

            jaccard_mean.update(valid_out, valid_mask)
            precision_mean.update(valid_out, valid_mask)
            recall_mean.update(valid_out, valid_mask)
            accuracy_mean.update(valid_out, valid_mask)

            jaccard_weighted.update(valid_out, valid_mask)
            precision_weighted.update(valid_out, valid_mask)
            recall_weighted.update(valid_out, valid_mask)
            accuracy_weighted.update(valid_out, valid_mask)
            
    # Print results
    print(f'Overall IoU: {jaccard.compute()*100:.2f}%')
    print(f'Mean IoU: {jaccard_mean.compute()*100:.2f}%')
    print(f'Weighted IoU: {jaccard_weighted.compute()*100:.2f}%')

    print(f'Overall Precision: {precision.compute()*100:.2f}%')
    print(f'Mean Precision: {precision_mean.compute()*100:.2f}%')
    print(f'Weighted Precision: {precision_weighted.compute()*100:.2f}%')

    print(f'Overall Recall: {recall.compute()*100:.2f}%')
    print(f'Mean Recall: {recall_mean.compute()*100:.2f}%')
    print(f'Weighted Recall: {recall_weighted.compute()*100:.2f}%')

    print(f'Overall Accuracy: {accuracy.compute()*100:.2f}%')
    print(f'Mean Accuracy: {accuracy_mean.compute()*100:.2f}%')
    print(f'Weighted Accuracy: {accuracy_weighted.compute()*100:.2f}%')

    if output_file is not None:
        with open(output_file,'w') as f:
            with redirect_stdout(f):
                print(f'Overall IoU: {jaccard.compute() * 100:.2f}%')
                print(f'Mean IoU: {jaccard_mean.compute() * 100:.2f}%')
                print(f'Weighted IoU: {jaccard_weighted.compute() * 100:.2f}%')

                print(f'Overall Precision: {precision.compute() * 100:.2f}%')
                print(f'Mean Precision: {precision_mean.compute() * 100:.2f}%')
                print(f'Weighted Precision: {precision_weighted.compute() * 100:.2f}%')

                print(f'Overall Recall: {recall.compute() * 100:.2f}%')
                print(f'Mean Recall: {recall_mean.compute() * 100:.2f}%')
                print(f'Weighted Recall: {recall_weighted.compute() * 100:.2f}%')

                print(f'Overall Accuracy: {accuracy.compute() * 100:.2f}%')
                print(f'Mean Accuracy: {accuracy_mean.compute() * 100:.2f}%')
                print(f'Weighted Accuracy: {accuracy_weighted.compute() * 100:.2f}%')

    return accuracy.compute(),jaccard_mean.compute()


