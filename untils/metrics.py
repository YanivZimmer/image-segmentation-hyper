import torch
from torchmetrics.classification import MulticlassJaccardIndex
NUM_OF_CLASS=11
# def calculate_iou(pred_mask,gt_mask):
#     intersection = torch.logical_and(gt_mask, pred_mask)
#     union = torch.logical_or(gt_mask, pred_mask)
#     iou_score = torch.sum(intersection).float() / torch.sum(union).float()
#     return iou_score


class IouCalculator:
    def __init__(self, num_classes=NUM_OF_CLASS, device="cuda"):
        self.metric = MulticlassJaccardIndex(
            num_classes=num_classes, ignore_index=0
        ).to(device)

    def calculate_iou(self, pred_mask, gt_mask, ignore=(0,)):
        pred_mask=torch.argmax(pred_mask,dim=1)
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
class MulticlassDiceLoss(torch.nn.Module):
    """Reference: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss
    """
    def __init__(self, num_classes, softmax_dim=None):
        super().__init__()
        self.num_classes = num_classes
        self.softmax_dim = softmax_dim

    def forward(self, logits, targets, reduction='mean', smooth=1e-6):
        """The "reduction" argument is ignored. This method computes the dice
        loss for all classes and provides an overall weighted loss.
        """
        print(logits.shape)
        print(targets.shape)
        probabilities = logits
        if self.softmax_dim is not None:
            probabilities = torch.nn.Softmax(dim=self.softmax_dim)(logits)
        # end if
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes)
        print(targets_one_hot.shape)
        # Convert from NHWC to NCHW
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)
        
        # Exclude the background class (label 0)
        probabilities = probabilities[:, 1:]  # Exclude the first channel (background)
        targets_one_hot = targets_one_hot[:, 1:]  # Exclude the first channel (background)

        # Multiply one-hot encoded ground truth labels with the probabilities to get the
        # predicted probability for the actual class.
        intersection = (targets_one_hot * probabilities).sum(dim=(2, 3))
        
        mod_a = targets_one_hot.sum(dim=(2, 3))
        mod_b = probabilities.sum(dim=(2, 3))
        
        dice_coefficient = (2. * intersection + smooth) / (mod_a + mod_b + smooth)
        dice_loss = 1 - dice_coefficient.mean(dim=1)  # Dice loss for each class, averaged over batch
        
        return dice_loss.mean()  # Final loss, averaged over all classes and batch

# end class MulticlassDiceLoss

#criterion = MulticlassDiceLoss(num_classes=3, softmax_dim=1)
#y = torch.randn(10, 3, 4, 4)
#ground_truth = torch.randint(0, 3, (10, 4, 4))
#print(y.shape)
#criterion(y, ground_truth)