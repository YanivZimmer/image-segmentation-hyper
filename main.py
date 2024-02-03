import os

from untils.data_split import DataSplit

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import torchvision.transforms
from torch.utils.data import DataLoader

from data_loaders.segmentation_dataset import SegmentationDataset
from models.big_unet import UNet
from untils.loss import MixedLoss
from untils.test import evaluate, metric
from untils.train import EarlyStopping, train_one_epoch

EPOCHS = 2
TRAIN_MODEL = True
EVALUATE = True
N_CLASS = 11

# LEARNING_RATE = 0.00001 acc 59%
# LEARNING_RATE = 0.0000025
# LEARNING_RATE = 0.000005
LEARNING_RATE = 0.000015


def main():
    criterion = MixedLoss(0, 2.0, "cuda")
    es = EarlyStopping(patience=2, mode="max")
    # SIZE = 572
    model = UNet(25, 11, band_selection=False)
    ds = DataSplit()
    test, val, train = ds.get_files("./assets")

    data_path = (
        "/media/orange/i_want_to_add_to/Datasets/HS_Drive_v2/Image_dataset/cubes_fl32"
    )
    labels_path = (
        "/media/orange/i_want_to_add_to/Datasets/HS_Drive_v2/Image_dataset/labels"
    )

    train_dataset = SegmentationDataset(
        image_dir=data_path,
        n_class=N_CLASS,
        mask_dir=labels_path,
        mode="Hyper",
        data_key="cube",
        samples_names=train,
        transform=None,
    )  # torchvision.transforms.Resize((SIZE, SIZE)))
    val_dataset = SegmentationDataset(
        image_dir=data_path,
        n_class=N_CLASS,
        mask_dir=labels_path,
        mode="Hyper",
        data_key="cube",
        samples_names=val,
        transform=None,
    )  # torchvision.transforms.Resize((SIZE, SIZE)))
    test_dataset = SegmentationDataset(
        image_dir=data_path,
        n_class=N_CLASS,
        mask_dir=labels_path,
        mode="Hyper",
        data_key="cube",
        samples_names=test,
        transform=None,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    if TRAIN_MODEL:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[15, 55, 105], gamma=0.75
        )
        for epoch in range(EPOCHS):
            loss = train_one_epoch(train_dataloader, model, optimizer, criterion)
            # train_acc, train_iou = evaluate(train_dataloader, model, metric=metric)
            val_acc, val_iou = evaluate(val_dataloader, model, metric=metric)
            scheduler.step()
            print(
                f"EPOCH: {epoch}, TRAIN LOSS: {loss}, VAL Acc: {val_acc}, VAL iou: {val_iou} "
            )
            if es.early_stop:
                print("\n\n -------------- EARLY STOPPING -------------- \n\n")
                break
    if EVALUATE:
        test_score = evaluate(test_dataloader, model, metric=metric)
        print(f"Test score: {test_score}")


def split_data():
    ds = DataSplit()
    test, val, train = ds.get_files("./assets")
    train_data = (
        "/media/orange/i_want_to_add_to/Datasets/HS_Drive_v2/Image_dataset/cubes_fl32"
    )
    train_labels = (
        "/media/orange/i_want_to_add_to/Datasets/HS_Drive_v2/Image_dataset/labels"
    )
    train_dataset = SegmentationDataset(
        image_dir=train_data,
        n_class=N_CLASS,
        mask_dir=train_labels,
        mode="Hyper",
        data_key="cube",
        samples_names=train,
        transform=None,
    )
    print(len(train_dataset))


if __name__ == "__main__":
    main()
