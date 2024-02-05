import os
from typing import List, Set, Iterable

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
    def __init__(
        self,
        image_dir,
        mask_dir,
        n_class,
        mode,
        data_key,
        samples_names: Iterable[str],
        transform=None,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.n_class = n_class
        self.transform = transform
        self.mode = mode
        self.data_key = data_key
        self.samples_names = set(samples_names)
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.masks = list(self.samples_names.intersection(self.masks))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.masks)

    def file_to_mat(self, filename: str, key: str) -> np.ndarray:
        if filename.endswith("png"):
            return png_to_array(filename)
        return loadmat(filename)[key]

    def image_opener_mat(self, path):
        return torch.from_numpy(
            self.file_to_mat(path, self.data_key)
        )  # [:2,:,:]#.unsqueeze(0)

    def image_opener(self, path):
        image = self.to_tensor(Image.open(path).convert("RGB"))
        if self.transform:
            image = self.transform(image)
        return image

    def mask_to_categorcial(self, mask):
        return torch.nn.functional.one_hot(mask.to(torch.int64), self.n_class)

    def mask_opener(self, path):
        mask = torch.from_numpy(np.array(Image.open(path)))
        if self.transform:
            mask = self.transform(mask)
        return mask.to(torch.int64)  # self.mask_to_categorcial(mask)

    def get_data_name(self, lablefile):
        base_name = os.path.splitext(os.path.basename(lablefile))[0]
        data_files = list(filter(lambda a: a.startswith(base_name), self.images))
        if len(data_files) == 0:
            raise AttributeError(f"no data for {base_name}")
        return data_files[0]
    def mask_to_other(self,tensor):
      preset_dict={1:1,2:1,3:2,4:2,5:2,6:2,7:2,8:2,9:2,10:2}
      mapped_values = torch.tensor([preset_dict.get(val.item(), 0) for val in tensor.flatten()])
      # Reshape the mapped values tensor to match the original tensor shape
      mapped_tensor = mapped_values.reshape(tensor.shape)
      return mapped_tensor

    def __getitem__(self, idx):
        mask_name = os.path.join(self.mask_dir, self.masks[idx])
        img_name = os.path.join(self.image_dir, self.get_data_name(self.masks[idx]))
        mask = self.mask_opener(mask_name)
        mask = self.mask_to_other(mask)
        # mask = self.to_tensor(Image.open(mask_name).convert('L'))#.unsqueeze(0)  # L mode for single-channel masks
        if self.mode == "RGB":
            image = self.image_opener(img_name)
        else:
            image = self.image_opener_mat(img_name)
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return {"image": image, "mask": mask}
