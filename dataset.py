import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class LspineDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        
        if self.mask_dir!=None:
            mask_path = os.path.join(self.mask_dir, self.masks[index])
            mask=np.load(mask_path)
        
        if self.transform is not None:
            if self.mask_dir!=None:
                augmentations = self.transform(image=image, mask=mask)
            else:
                augmentations = self.transform(image=image)
            image = augmentations["image"]


            if self.mask_dir!=None:
                mask = augmentations["mask"]

        if self.mask_dir!=None:
            return image, mask
        else:
            return image

class TestDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        w, h, c= image.shape
        if self.mask_dir!=None:
            mask_path = os.path.join(self.mask_dir, self.masks[index])
            mask=np.load(mask_path)
            mask[mask==255.0]=1.0
        
        if self.transform is not None:
            if self.mask_dir!=None:
                augmentations = self.transform(image=image, mask=mask)
            else:
                augmentations = self.transform(image=image)
            image = augmentations["image"]


            if self.mask_dir!=None:
                mask = augmentations["mask"]

        
        return image, mask, self.images[index], w, h