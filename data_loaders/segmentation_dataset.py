import os
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
def png_to_array(path: str) -> np.ndarray:
    image = Image.open(path)
    arr = np.asarray(image)
    return arr

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir,mode,data_key,transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mode = mode
        self.data_key= data_key
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.to_tensor = transforms.ToTensor()
    def __len__(self):
        return len(self.images)

    def file_to_mat(self, filename: str, key: str) -> np.ndarray:
        if filename.endswith("png"):
            return png_to_array(filename)
        return loadmat(filename)[key]

    def image_opener_mat(self, path):
        return torch.from_numpy(self.file_to_mat(path,self.data_key))#[:3,:,:]#.unsqueeze(0)

    def image_opener(self, path):
        return self.to_tensor(Image.open(path).convert('RGB'))

    def get_data_name(self,lablefile):
        base_name = os.path.splitext(os.path.basename(lablefile))[0]
        data_files = list(filter(lambda a: a.startswith(base_name), self.images))
        if len(data_files) == 0:
            raise AttributeError(f"no data for {base_name}")
        return data_files[0]
    def __getitem__(self, idx):
        mask_name = os.path.join(self.mask_dir, self.masks[idx])
        img_name = os.path.join(self.image_dir, self.get_data_name(self.masks[idx]))

        mask = self.to_tensor(Image.open(mask_name).convert('L'))#.unsqueeze(0)  # L mode for single-channel masks
        if self.mode =="RGB":
            image = self.image_opener(img_name)
        else:     
            image = self.image_opener_mat(img_name)
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return {'image':image, 'mask':mask}
