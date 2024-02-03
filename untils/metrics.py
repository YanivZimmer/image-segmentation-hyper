import torch
from torchmetrics.classification import MulticlassJaccardIndex

# def calculate_iou(pred_mask,gt_mask):
#     intersection = torch.logical_and(gt_mask, pred_mask)
#     union = torch.logical_or(gt_mask, pred_mask)
#     iou_score = torch.sum(intersection).float() / torch.sum(union).float()
#     return iou_score


class IouCalculator:
    def __init__(self, num_classes=11, device="cuda"):
        self.metric = MulticlassJaccardIndex(
            num_classes=num_classes, ignore_index=0
        ).to(device)

    def calculate_iou(self, pred_mask, gt_mask, ignore=(0,)):
        res = self.metric(pred_mask, gt_mask)
        return res

    # # Create masks for ignored labels
    # ignore_gt_mask = torch.zeros_like(gt_mask)
    # ignore_pred_mask = torch.zeros_like(pred_mask)
    # for label in ignore:
    #     ignore_gt_mask[gt_mask == label] = 1
    #     ignore_pred_mask[pred_mask == label] = 1
    #
    # # Exclude ignored labels from intersection and union
    # intersection = torch.logical_and(gt_mask * (1 - ignore_gt_mask), pred_mask * (1 - ignore_pred_mask))
    # union = torch.logical_or(gt_mask * (1 - ignore_gt_mask), pred_mask * (1 - ignore_pred_mask))
    #
    # # Calculate IoU score
    # iou_score = torch.sum(intersection).float() / torch.sum(union).float()
    #
    # return iou_score
