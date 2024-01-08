import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import torchvision.transforms
from torch.utils.data import DataLoader

from data_loaders.segmentation_dataset import SegmentationDataset
from models.big_unet import UNet
from untils.loss import MixedLoss
from untils.test import evaluate, metric
from untils.train import EarlyStopping, train_one_epoch

EPOCHS = 25
TRAIN_MODEL = True
EVALUATE = True
#LEARNING_RATE = 0.00001 acc 59%
LEARNING_RATE = 0.0000025
criterion = MixedLoss(0, 2.0)
es = EarlyStopping(patience=2, mode='max')
N_CLASS=11
#SIZE = 572
model = UNet(25,11)

train_data = '/media/orange/i_want_to_add_to/Datasets/HS_Drive_v2/Image_dataset/cubes_fl32'
#'/home/orange/datasets/hsi_drive_v2/train/data'
train_labels = '/media/orange/i_want_to_add_to/Datasets/HS_Drive_v2/Image_dataset/labels'
#'/home/orange/datasets/hsi_drive_v2/train/labels'
val_data = '/media/orange/i_want_to_add_to/Datasets/HS_Drive_v2/Image_dataset/cubes_fl32'
#'/home/orange/datasets/hsi_drive_v2/validation/data'
val_labels = '/media/orange/i_want_to_add_to/Datasets/HS_Drive_v2/Image_dataset/labels'
#'/home/orange/datasets/hsi_drive_v2/validation/labels'

train_dataset = SegmentationDataset(image_dir=train_data, n_class=N_CLASS, mask_dir=train_labels,
                                       mode="Hyper", data_key='cube',
                                       transform=None)#torchvision.transforms.Resize((SIZE, SIZE)))
val_dataset = SegmentationDataset(image_dir=val_data, n_class=N_CLASS, mask_dir=val_labels,
                                     mode="Hyper", data_key='cube',
                                     transform=None)#torchvision.transforms.Resize((SIZE, SIZE)))

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

if TRAIN_MODEL:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[3, 5, 6, 7, 8, 9, 10, 11, 13, 15], gamma=0.75)
    for epoch in range(EPOCHS):
        loss = train_one_epoch(train_dataloader, model, optimizer, criterion)
        dice = evaluate(val_dataloader, model, metric=metric)
        scheduler.step()
        print(f"EPOCH: {epoch}, TRAIN LOSS: {loss}, VAL DICE: {dice}")
        # es(dice, model, model_path=f"../data/bst_model{IMG_SIZE}_fold{FOLD_ID}_{np.round(dice,4)}.bin")
        # best_model = f"../data/bst_model{IMG_SIZE}__fold{FOLD_ID}_{np.round(es.best_score,4)}.bin"
        if es.early_stop:
            print('\n\n -------------- EARLY STOPPING -------------- \n\n')
            break
if EVALUATE:
    valid_score = evaluate(val_dataloader, model, metric=metric)
    print(f"Valid dice score: {valid_score}")
