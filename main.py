import math
import os
import uuid
from dog import LDoG
from dog import PolynomialDecayAverager
from untils.data_split import DataSplit
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234567)
import random
random.seed(7654321)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
import torch
torch.manual_seed(1010101)
import torchvision.transforms
from torch.utils.data import DataLoader
import statistics
from data_loaders.segmentation_dataset import SegmentationDataset
#from models.big_unet import UNet,SimpleFCNBS
from models.org_unet import UNet
from models.unet3d import UNet3D
from models.unet3d_v3 import Unet3dSlim
from untils.loss import MixedLoss
from untils.test import evaluate, metric, evaluate_old
from untils.train import EarlyStopping, train_one_epoch
import math
import sys
from pathlib import Path

#PATH="weights/checkpoint_bsnet"
EPOCHS =200#165+101#15#70#15#19
TRAIN_MODEL = True
EVALUATE = True
N_CLASS = 11
N_CLASS = 4
N_BANDS = 25
#MASK =None#[1,2,3,4,5]#None#[2,7,24,11,1]#None#[11 , 3, 13,  6 , 9]#None#[1,11,16,19,23] #BSNETS [11 , 3, 13,  6 , 9] #SNMF[18,11,3,4,0] #SPABS [2,7,24,11,1]# Gene  [17, 13,  4,  1, 16]
# we sould get > 0.56 for 7
#bsnets  [11, 3, 13] snmf [24, 21, 8] spabs [11 ,22, 24] 
#bsnets [4, 13, 11, 24, 3, 2, 1] 7 snmf [5, 1, 16, 22, 8, 11, 4] spabs [4, 13, 11, 24, 3, 2, 1]
#snmf [5, 1, 16, 22,23,24, 8, 11, 4] spabs [4, 13, 11, 24, 3, 2, 1,22,20]
MASK = None#[11, 3, 13]#[4, 13, 11, 24, 3]#[ 1,  6, 20,  7, 13]#[11 , 3, 13]#[11, 19, 6, 20, 9]#[11, 19, 6, 20, 9]#[10, 11, 13, 15, 19, 24]  #bsnets[11, 19, 6, 20, 9, 15] #ehbs[10, 11, 13, 15, 19, 24]
#[11 , 9 , 6]
#[1, 5, 10, 13, 17, 20]#... #[16, 3, 0, 6, 18, 8]
# [11  ,9 , 6 , 7  ,5 ,15]


#[24, 21, 8] #[5, 1, 16, 22, 8, 11, 4]#[11 , 3, 13,  6 , 9]#[11 , 3, 13,  6 , 9]# [11 , 3, 13,  6 , 9]#[11, 3, 13, 7, 6, 9, 19, 8, 12] #[5, 1, 16, 22,23,24, 8, 11, 4]#[2,7,24,11,1]#[20, 18, 10,  8,  1]#[22, 20, 13, 11,  8,  1,  0][23, 20, 16, 13, 10,  8,  4,  1,  0]
#[4, 13, 11, 24, 3, 2, 1]  
#[2,7,24,11,1]#[4, 13, 11, 24, 3, 2, 1]#[11, 3, 13, 7, 6, 9, 19, 8, 12, 4]#    [11, 22, 24]
#[4, 13, 11, 24, 3, 2, 1]
TYPE=("experimental_gumbelones_3_class")#gumbelones"#"seedv2_gumbel_85_0999"#high_ones-noise05_65+101epc_tmp1.5alpha0.999"
#None#[11, 3, 13, 7, 6, 9, 19]
# [11, 3, 13, 7, 6, 9, 19]#None#[  8 ,24, 11,  7,  5 , 4  ,3 , 6,  2,  1]  # [5, 1, 16, 22, 8, 11, 4]
#4
#snmf [19, 10, 21, 8] [21, 15] bs  [11 19  9  6] [11 19  6 20]

#MASK=range(N_BANDS)
#TYPE="all"

print("MASK",MASK)
N_TARGET_BANDS = len(MASK) if MASK is not None else int(sys.argv[1])
# LEARNING_RATE = 0.00001 acc 59%
# LEARNING_RATE = 0.0000025
# LEARNING_RATE = 0.000005
LEARNING_RATE = 0.00005 # 3 labels
#LEARNING_RATE = 0.000025
#LEARNING_RATE = 0.0001 best for all labels and 60 epc

#LEARNING_RATE = 0.0001 #best for all labels and 70 epc
LEARNING_RATE= 0.0001
LEARNING_RATE=  0.005#0.001
BAND_SELECTION = True if MASK is None else False



print("MASK ", MASK, TYPE, "N_TARGET_BANDS",N_TARGET_BANDS,"BAND_SELECTION",BAND_SELECTION)



def plot_val_score(validation_scores,bands,metric="oa",step_size=10):
    try:
        # Assuming you have a list of validation scores and corresponding epochs
        epochs = list(range(0, len(validation_scores)*10, 10))
        print(epochs)

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, validation_scores, marker='o', linestyle='-', color='b')
        plt.title(f'Validation Score per 10 Epochs bands {bands}')
        plt.xlabel('Epochs')
        plt.ylabel('Validation Score')
        plt.grid(True)
        #plt.yticks(range(5, 100, 5))
        # Save the plot
        plt.savefig(f'plots/{TYPE}-{N_TARGET_BANDS}-{metric}-validation_scores_plot-{uuid.uuid4().hex}.png')
    except Exception as e:
        with open(f'plots/{TYPE}-{N_TARGET_BANDS}-{metric}-validation_scores_plot-{uuid.uuid4().hex}.txt','w') as f:
            f.write(f"{validation_scores}")

def save_checkpoint(model, model_path):
    model_path = Path(model_path)
    parent = model_path.parent
    os.makedirs(parent, exist_ok=True)
    torch.save(model.state_dict(), model_path)

def main(lr,lr_factor):
    #criterion = MixedLoss(0, 2.0, "cuda")

    es = EarlyStopping(patience=2, mode="max")
    # SIZE = 572
    model =  UNet(25, N_TARGET_BANDS, N_CLASS, band_selection=BAND_SELECTION,mask=MASK)
    #model = SimpleFCNBS(N_TARGET_BANDS, N_CLASS).to("cuda")
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
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)#was 4 now 16-18.6.2024

    # Calculate class frequencies
    num_classes = N_CLASS
    class_counts = torch.zeros(num_classes, dtype=torch.int64)

    # Count the frequencies of each class in the dataset
    for data in train_dataloader:
        #import pdb; pdb.set_trace()
        labels = np.array(data["mask"]).flatten()
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            if u != 0:  # Ignore label 0
                class_counts[u] += c

    # Calculate class frequencies
    class_freqs = class_counts.float() / class_counts.sum()

    # Calculate inverse-frequency weights
    weights = 1.0 / class_freqs
    non_nan_mask = ~torch.isnan(weights)
    weights = weights /weights[1:].sum()#torch.sum(weights[non_nan_mask])  # normalize to sum to 1
    weights= weights.to("cuda")
    # Print class frequencies and weights
    print(f"Class frequencies: {class_freqs}")
    print(f"Class weights: {weights}")
    criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=0)
#
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
    print(train_dataset[0]['image'].shape)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    val_test_dataset=torch.utils.data.ConcatDataset([test_dataset, val_dataset])
    val_test_dataloader = DataLoader(val_test_dataset, batch_size=4, shuffle=True)
    averager = None#PolynomialDecayAverager(model)
    if TRAIN_MODEL:
        #lr = LEARNING_RATE
        #modified_lr = [
        #  {"params": list(model.parameters())[1:], "lr": lr},
        #  {"params": list(model.parameters())[:1], "lr": lr_factor * lr},
        #]
        #optimizer = torch.optim.Adam(modified_lr, lr=lr, weight_decay=0.0001)
        print("AdamMod")
        #optimizer = LDoG(model.parameters())#, reps_rel=1e-6)#
        #averager = PolynomialDecayAverager(net)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #    optimizer, milestones=[3,5,15,35, 55, 105], gamma=0.75#
        #)
        selected_mask=None
        losses=[]
        patience = True
        val_accs=[]
        val_ious=[]
#        network2_weights = torch.load(PATH)
        
        # Get the state dictionary of the model
#        model_state_dict = model.state_dict()
        
        # Print the model's layers
#        print("Model layers:")
#        for name, param in model_state_dict.items():
#            print(name)
        
        # Print Network2 weights
#        print("\nNetwork2 weights:")
#        for name, param in network2_weights.items():
#            print(name)
        
        # Update only the common layers and rest layers in the model with weights from Network2
 #       for name, param in network2_weights.items():
 #           if name in model_state_dict:
 #               model_state_dict[name].copy_(param)
        
        # Load the updated state dictionary back into the model
  #      model.load_state_dict(model_state_dict)
        checkpoint_file=f"weights/unet-{N_TARGET_BANDS}-{TYPE}"
        for epoch in range(EPOCHS):
            #if epoch ==80:
            #  torch.save(model.state_dict(), PATH)
            #  break
            print(f"Epoch {epoch} out of {EPOCHS}")
            loss = train_one_epoch(train_dataloader, model, optimizer,averager, criterion,lam=0.25)#lam=0.25 for 7 with sigma 0.5
            losses.append(loss)
            if model.band_selection:
              #print(model.ehbs.get_gates("prob"))
              print(model.ehbs.get_gates("raw"))
              #print(type(model.ehbs.get_gates("raw")[0]))
              #selected_mask = model.ehbs.get_topk_stable(torch.Tensor(model.ehbs.get_gates("raw")),N_TARGET_BANDS)
              selected_mask = torch.argmax(torch.from_numpy(model.ehbs.get_gates("raw")[0]),dim=1)
              print(selected_mask)
              #model = UNet(25, N_TARGET_BANDS, N_CLASS, band_selection=False,mask=selected_mask)
            # train_acc, train_iou = evaluate(train_dataloader, model, metric=metric)
            #selected_mask = torch.argmax(torch.from_numpy(model.ehbs.get_gates("raw")[0]),dim=1)
            print(selected_mask)
            if epoch % 10 ==0:
                val_acc, val_iou = evaluate_old(val_dataloader, model, metric=metric)
                _ = evaluate(val_test_dataloader, model,N_CLASS)  # metric=metric)
                if len(val_accs)==0 or val_accs[-1]<val_acc:
                    print(f"oa improved. {val_acc}")
                    save_checkpoint(model,model_path=checkpoint_file)
                else:
                    print(f"oa not improved! {val_acc} prev was {val_accs[-1]}")
                val_accs.append(val_acc.item())
                val_ious.append(val_iou.item())
            #if epoch == EPOCHS-165 and model.band_selection:
            #  model = UNet(25, N_TARGET_BANDS, N_CLASS, band_selection=False,mask=selected_mask)
            #  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

            # if averager is None:
            #   scheduler.step()
            print(
                f"EPOCH: {epoch} TYPE {TYPE} TRAIN LOSS: {loss}, VAL Acc: {val_acc}, VAL iou: {val_iou} "
            )
            
            #if len(losses)>125 and loss>=losses[-1] and loss>=losses[-2]:
            #    if patience:
            #        patience= False
            #    else:
            #        break
            #else:
            #    patience = True
            if es.early_stop:
                print("\n\n -------------- EARLY STOPPING -------------- \n\n")
                break
        plot_val_score(val_accs,selected_mask)
        plot_val_score(val_ious,selected_mask,"iou")
    if EVALUATE:
        model.load_state_dict(torch.load(checkpoint_file))
        model.train(mode=False)
        test_val_score = evaluate(val_test_dataloader,model,N_CLASS)# metric=metric)
        train_score = evaluate(train_dataloader,model, N_CLASS)#metric=metric)
        test_score = evaluate(test_dataloader,model, N_CLASS)#metric=metric)
        print(f"Test score: {test_score}")
        return test_score,val_acc, val_iou,train_score


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
    N_REPEAT=1
    #for lr_factor in [1,2,4,8,16]:
    #  for lr in [0.001,0.00015,0.00005,0.00001,0.000005,0.0000001]:
    #    print("next is",lr,lr_factor )     
    lr = LEARNING_RATE
    lr_factor = 1#-math.log(lr)
    accs=[]
    ious=[]
    ious_val=[]
    accs_val=[]
    
    ious_train=[]
    accs_train=[]
    for i in range(N_REPEAT):
        print("iter=",i)
        (acc,iou),val_acc,val_iou,(acc_train,iou_train)=main(lr=lr,lr_factor=lr_factor)
        accs.append(acc.item())
        ious.append(iou.item())
        accs_val.append(val_acc.item())
        ious_val.append(val_iou.item())
        ious_train.append(iou_train.item())
        accs_train.append(acc_train.item())
        print("MASK ", MASK, TYPE ,"N_TARGET_BANDS", N_TARGET_BANDS, "BAND_SELECTION", BAND_SELECTION, "LEARNING_RATE",
              LEARNING_RATE)
        if i>0:
            print("OA of all", accs, "mean", statistics.fmean(accs), "std", statistics.stdev(accs))
            print("IOU of all", ious, "mean", statistics.fmean(ious), "std", statistics.stdev(ious))
            print("val OA of all" ,accs_val, "mean",statistics.fmean(accs_val), "std", statistics.stdev(accs_val))
            print("val IOU of all" ,ious_val,  "mean",statistics.fmean(ious_val),"std", statistics.stdev(ious_val))
            print("train OA of all" ,accs_train, "mean",statistics.fmean(accs_train), "std", statistics.stdev(accs_train))
            print("train IOU of all" ,ious_train,  "mean",statistics.fmean(ious_train),"std", statistics.stdev(ious_train))
    with open(f"logs/unetv0-{N_TARGET_BANDS}-{TYPE}-{uuid.uuid4().hex}.txt", 'w') as f:
        f.writelines([
            f"MASK {str(MASK)} N_TARGET_BANDS {N_TARGET_BANDS} BAND_SELECTION {BAND_SELECTION}\n"
            f" LEARNING_RATE {LEARNING_RATE} OA of all {accs}\n"
            f"mean {statistics.fmean(accs)} std {statistics.stdev(accs)}\n",
        f"IOU of all {ious} mean {statistics.fmean(ious)} std {statistics.stdev(ious)}\n",
        f"val OA of all {accs_val} mean {statistics.fmean(accs_val)} std {statistics.stdev(accs_val)}\n",
        f"val IOU of all {ious_val} mean {statistics.fmean(ious_val)} std{statistics.stdev(ious_val)}\n",
        f"train OA of all {accs_train} mean {statistics.fmean(accs_train)} std {statistics.stdev(accs_train)}\n",
        f"train IOU of all {ious_train} mean {statistics.fmean(ious_train)} std {statistics.stdev(ious_train)}\n"])
    #
    # import uuid
    # with open(f"temp-{uuid.uuid4().hex}",'w') as f:
    #     f.write("MASK ", MASK, "N_TARGET_BANDS", N_TARGET_BANDS, "BAND_SELECTION", BAND_SELECTION, "LEARNING_RATE",
    #           LEARNING_RATE)
    #     f.write("OA of all", accs, "mean", statistics.fmean(accs), "std", statistics.stdev(accs))
    #     f.write("IOU of all", ious, "mean", statistics.fmean(ious), "std", statistics.stdev(ious))
    #     f.write("val OA of all", accs_val, "mean", statistics.fmean(accs_val), "std", statistics.stdev(accs_val))
    #     f.write("val IOU of all", ious_val, "mean", statistics.fmean(ious_val), "std", statistics.stdev(ious_val))
    #     f.write("train OA of all", accs_train, "mean", statistics.fmean(accs_train), "std",
    #           statistics.stdev(accs_train))
    #     f.write("train IOU of all", ious_train, "mean", statistics.fmean(ious_train), "std",
    #           statistics.stdev(ious_train))
    print("MASK ", MASK, "N_TARGET_BANDS", N_TARGET_BANDS, "BAND_SELECTION", BAND_SELECTION, "LEARNING_RATE", LEARNING_RATE)
    print("OA of all" ,accs, "mean",statistics.fmean(accs), "std", statistics.stdev(accs))
    print("IOU of all" ,ious,  "mean",statistics.fmean(ious),"std", statistics.stdev(ious))
    print("val OA of all" ,accs_val, "mean",statistics.fmean(accs_val), "std", statistics.stdev(accs_val))
    print("val IOU of all" ,ious_val,  "mean",statistics.fmean(ious_val),"std", statistics.stdev(ious_val))
    print("train OA of all" ,accs_train, "mean",statistics.fmean(accs_train), "std", statistics.stdev(accs_train))
    print("train IOU of all" ,ious_train,  "mean",statistics.fmean(ious_train),"std", statistics.stdev(ious_train))
    #print("it was",lr,lr_factor )
