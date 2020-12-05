from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
import scipy.io as sio
import re
import torch
import os


class BudleBay(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=True):

        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []
        self.labels = []
        self.download = download

        if self.download:

            imgs_list = os.listdir(self.root)

            for img_file in imgs_list:
                if img_file.endswith('.jpg'):
                    self.samples.append(img_file)

    def __getitem__(self, index):

        img = self.samples[index]

        img_path = os.path.join(self.root, img)

        im = Image.open(img_path)

        pix = np.array(im)
        pix = pix[:, :, 0:3]

        im = Image.fromarray(np.uint8(pix))

        if self.transform is not None:
            im = self.transform(im)

        return im, img_path

    def __len__(self):
        return len(self.samples)
