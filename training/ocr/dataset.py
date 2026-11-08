import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


class CropDataset(Dataset):
    def __init__(self, crops_dir, char_map, img_h=32, img_w=320, max_len=25):
        self.crops = []
        self.char_map = char_map
        self.rev = {v: k for k, v in char_map.items()}
        self.img_h = img_h
        self.img_w = img_w
        self.max_len = max_len
        for root, dirs, files in os.walk(crops_dir):
            for fn in files:
                if fn.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.crops.append(os.path.join(root, fn))

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        p = self.crops[idx]
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.img_w, self.img_h))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)
        return torch.from_numpy(img), p
