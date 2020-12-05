from torch.utils.data.dataset import Dataset
from PIL import Image
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

        img_files = os.path.join(self.root)

        self.img_file_dir = os.path.join(img_files, self.split, 'imgs')
        self.lab_file_dir = os.path.join(img_files, self.split, 'labels')

        if self.download:

            imgs_list = os.listdir(self.img_file_dir)
            lab_list = os.listdir(self.lab_file_dir)

            for img_file in imgs_list:
                img_id = ''.join([i for i in img_file if i.isdigit()])
                self.samples.append((img_file, img_id))

            for lab_file in lab_list:
                lab_id = ''.join([i for i in lab_file if i.isdigit()])
                self.labels.append((lab_file, lab_id))

            out_file = 'samples_bb_' + self.split + '.txt'
            out_lab = 'labels_bb_' + self.split + '.txt'

            with open(out_file, 'w') as fp:
                fp.write('\n'.join('{} {}'.format(x[0], x[1]) for x in self.samples))

            with open(out_lab, 'w') as f:
                f.write('\n'.join('{} {}'.format(x[0], x[1]) for x in self.labels))

        else:

            in_file = 'samples_bb_' + self.split + '.txt'
            with open(in_file, 'r') as fp:
                data = fp.readlines()
                data = [x.strip() for x in data]

            for line in data:
                str_list = []
                str_list = re.split(' ', line)
                self.samples.append((str_list[0], str_list[1]))

    def __getitem__(self, index):

        img, _ = self.samples[index]
        label, _ = self.labels[index]

        img_path = os.path.join(self.img_file_dir, img)
        label_path = os.path.join(self.lab_file_dir, label)

        im = sio.loadmat(img_path)
        lab = sio.loadmat(label_path)

        t_im = im['t_img']
        t_lab = torch.from_numpy(lab['l_img'])

        if self.transform is not None:
            t_im = self.transform(t_im)

        t_lab = t_lab.long()

        return t_im, t_lab

    def __len__(self):
        return len(self.samples)
