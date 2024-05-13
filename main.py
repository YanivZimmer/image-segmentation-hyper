import math
import os
from dog import LDoG
from dog import PolynomialDecayAverager
from untils.data_split import DataSplit

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
import torch
import torchvision.transforms
from torch.utils.data import DataLoader
import statistics
from data_loaders.segmentation_dataset import SegmentationDataset
from models.big_unet import UNet
from untils.loss import MixedLoss
from untils.test import evaluate, metric
from untils.train import EarlyStopping, train_one_epoch
import math
import sys
EPOCHS = 65#15#19
TRAIN_MODEL = True
EVALUATE = True
N_CLASS = 11
#N_CLASS = 4
N_BANDS = 25
#MASK =None#[1,2,3,4,5]#None#[2,7,24,11,1]#None#[11 , 3, 13,  6 , 9]#None#[1,11,16,19,23] #BSNETS [11 , 3, 13,  6 , 9] #SNMF[18,11,3,4,0] #SPABS [2,7,24,11,1]#
# we sould get > 0.56 for 7

MASK = None#[4, 13, 11, 24, 3, 2, 1]

#None#[11, 3, 13, 7, 6, 9, 19]
# [11, 3, 13, 7, 6, 9, 19]#None#[  8 ,24, 11,  7,  5 , 4  ,3 , 6,  2,  1]  # [5, 1, 16, 22, 8, 11, 4]


print("MASK",MASK)
N_TARGET_BANDS = len(MASK) if MASK is not None else int(sys.argv[1])
# LEARNING_RATE = 0.00001 acc 59%
# LEARNING_RATE = 0.0000025
# LEARNING_RATE = 0.000005
#LEARNING_RATE = 0.00005 # 3 labels
#LEARNING_RATE = 0.000025
#LEARNING_RATE = 0.0001 best for all labels and 60 epc

LEARNING_RATE = 0.0001

BAND_SELECTION = True if MASK is None else False
print("MASK ",MASK, "N_TARGET_BANDS",N_TARGET_BANDS,"BAND_SELECTION",BAND_SELECTION)
def main(lr,lr_factor):
    criterion = MixedLoss(0, 2.0, "cuda")
    es = EarlyStopping(patience=2, mode="max")
    # SIZE = 572
    model = UNet(25, N_TARGET_BANDS, N_CLASS, band_selection=BAND_SELECTION,mask=MASK)
    ds = DataSplit()
    test, val, train = ds.get_files("./assets")

    data_path = "/dsi/scratch/home/dsi/yanivz_datasets/HSI_Drive_v2_01/Image_dataset/cubes_fl32"#
        #"/cortex/data/images/hyperspectral/HS_Drive_v2/Image_dataset/cubes_fl32"
    #"/home/dsi/yanivz/data/HS_Drive_v2/Image_dataset/cubes_fl32"
        #"/media/orange/i_want_to_add_to/Datasets/HS_Drive_v2/Image_dataset/cubes_fl32"
    
    labels_path = "/dsi/scratch/home/dsi/yanivz_datasets/HSI_Drive_v2_01/Image_dataset/labels"
        #"/home/dsi/yanivz/data/HS_Drive_v2/Image_dataset/labels"
        #"/media/orange/i_want_to_add_to/Datasets/HS_Drive_v2/Image_dataset/labels"
    

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
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)
    averager = None#PolynomialDecayAverager(model)
    if TRAIN_MODEL:
        #lr = LEARNING_RATE
        modified_lr = [
          {"params": list(model.parameters())[1:], "lr": lr},
          {"params": list(model.parameters())[:1], "lr": lr_factor * lr},
        ]
        optimizer = torch.optim.Adam(modified_lr, lr=lr)
        print("AdamMod")
        #optimizer = LDoG(model.parameters())#, reps_rel=1e-6)#
        #averager = PolynomialDecayAverager(net)
        #optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #    optimizer, milestones=[3,5,15,35, 55, 105], gamma=0.75
        #)
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch} out of {EPOCHS}")
            loss = train_one_epoch(train_dataloader, model, optimizer,averager, criterion,lam=0.25)#lam=0.25 for 7 with sigma 0.5
            if model.band_selection:
              print(model.ehbs.get_gates("prob"))
              print(model.ehbs.get_gates("raw"))
            # train_acc, train_iou = evaluate(train_dataloader, model, metric=metric)
            val_acc, val_iou = evaluate(val_dataloader, model, metric=metric)
            # if averager is None:
            #   scheduler.step()
            print(
                f"EPOCH: {epoch}, TRAIN LOSS: {loss}, VAL Acc: {val_acc}, VAL iou: {val_iou} "
            )
            if es.early_stop:
                print("\n\n -------------- EARLY STOPPING -------------- \n\n")
                break
    if EVALUATE:
        test_score = evaluate(test_dataloader, model, metric=metric)
        print(f"Test score: {test_score}")
        return test_score


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
    N_REPEAT=3
    #for lr_factor in [1,2,4,8,16]:
    #  for lr in [0.001,0.00015,0.00005,0.00001,0.000005,0.0000001]:
    #    print("next is",lr,lr_factor )     
    lr = LEARNING_RATE
    lr_factor = 1#-math.log(lr)
    accs=[]
    ious=[]
    for i in range(N_REPEAT):
        print("iter=",i)
        acc,iou=main(lr=lr,lr_factor=lr_factor)
        accs.append(acc.item())
        ious.append(iou.item())
        print("MASK ", MASK, "N_TARGET_BANDS", N_TARGET_BANDS, "BAND_SELECTION", BAND_SELECTION, "LEARNING_RATE",
              LEARNING_RATE)
        if i>=2:
            print("OA of all", accs, "mean", statistics.fmean(accs), "std", statistics.stdev(accs))
            print("IOU of all", ious, "mean", statistics.fmean(ious), "std", statistics.stdev(ious))

    print("MASK ", MASK, "N_TARGET_BANDS", N_TARGET_BANDS, "BAND_SELECTION", BAND_SELECTION, "LEARNING_RATE", LEARNING_RATE)
    print("OA of all" ,accs, "mean",statistics.fmean(accs), "std", statistics.stdev(accs))
    print("IOU of all" ,ious,  "mean",statistics.fmean(ious),"std", statistics.stdev(ious))

    #print("it was",lr,lr_factor )
