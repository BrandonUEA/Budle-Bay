from torch.utils.data.dataset import Dataset
from PIL import Image
import scipy.io as sio
import re
import numpy as np
import os


class BudleBay(Dataset):
    def __init__(self, root, transform=None, target_transform=None, download=True):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []
        self.download = download

        if self.download:

            imgs_list = os.listdir(self.root)

            for img_file in imgs_list:
                if img_file.endswith('.mat'):
                    coords = list(map(int, re.findall(r'\d+', img_file)))
                    self.samples.append((img_file, coords))

            out_file = 'samples_bb_.txt'

            with open(out_file, 'w') as fp:
                fp.write('\n'.join('{} {}'.format(x[0], x[1]) for x in self.samples))

        else:

            in_file = 'samples_bb_.txt'
            with open(in_file, 'r') as fp:
                data = fp.readlines()
                data = [x.strip() for x in data]

            for line in data:
                str_list = []
                str_list = re.split(' ', line)
                self.samples.append((str_list[0], str_list[1]))


    def __getitem__(self, index):

        img, coords = self.samples[index]

        img_path = os.path.join(self.root, img)

        im = sio.loadmat(img_path)
        t_im = im['img']

        if self.transform is not None:
            im = self.transform(t_im)

        return im, img_path

    def __len__(self):
        return len(self.samples)
