from torch.utils.data.dataset import Dataset
import scipy.io as sio
import re
import os
import torch


class BudleBay(Dataset):
    def __init__(self, root, transform=None, target_transform=None):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []

        imgs_list = os.listdir(self.root)
        for img_file in imgs_list:
            if img_file.endswith('.mat'):
                coords = list(map(int, re.findall(r'\d+', img_file)))
                if img_file != 'bb_ms17.mat':
                    self.samples.append((img_file, coords))

    def __getitem__(self, index):

        img, coords = self.samples[index]

        img_path = os.path.join(self.root, img)

        im = sio.loadmat(img_path)
        t_im = im['img']

        # first transform to convert to double shape: [C x H x W] range: [0, 1]
        if self.transform is not None:
            t_im = self.transform(t_im)

            # pre-process
            r = t_im[0, ...]
            g = t_im[1, ...]
            b = t_im[2, ...]
            # nir = t_im[3, ...]

            # Normalised Difference Vegetation Index
            # ndvi = torch.div((nir - r), (nir + r)).unsqueeze(dim=0)
            # tensor(-0.7635)
            # tensor(0.7287)
            # ndvi = torch.div(ndvi - (-0.7635), 0.7287 - (-0.7635))

            # Atmospheric Resistant Vegetation Index
            # iavi84_a = (nir - (r - (torch.mul(0.84, (b - r)))))
            # iavi84_b = (nir + (r - (torch.mul(0.84, (b - r)))))
            # iavi84 = torch.div(iavi84_a, iavi84_b).unsqueeze(dim=0)
            # tensor(-0.8012)
            # tensor(2.1362)
            # iavi84 = torch.div(iavi84 - (-0.8012), 2.1362 - (-0.8012))

            # Modified Soil Adjusted Vegetation Index
            # msavi_a = torch.mul(2, nir) + 1
            # msavi_b = torch.mul(8, (nir-r))
            # msavi_c = torch.sqrt(torch.pow(msavi_a, 2) - msavi_b)
            # msavi = torch.mul(0.5, (msavi_a - msavi_c)).unsqueeze(dim=0)
            # tensor(-0.7108)
            # tensor(0.7305)
            # msavi = torch.div(msavi - (-0.7108), 0.7305 - (-0.7108))

            # Modified Chlorophyll Absorption Ratio Index
            # mcari_a = torch.mul(2, nir) + 1
            # mcari_b = torch.mul(2.5, (nir-r))
            # mcari_c = torch.mul(1.3, (nir-b))
            # mcari_d = (torch.mul(6, nir) - torch.mul(5, r))
            # mcari = torch.div(torch.mul(1.5, (mcari_b - mcari_c)), torch.sqrt(torch.pow(mcari_a, 2) - mcari_d - 0.5)).unsqueeze(dim=0)
            # tensor(-1.0109)
            # tensor(1.2145)
            # mcari = torch.div(mcari - (-1.0109), 1.2145 - (-1.0109))

            # Green Normalised Difference Vegetation Index
            # gndvi = torch.div((nir - g), (nir + g)).unsqueeze(dim=0)
            # tensor(-0.7209)
            # tensor(0.5954)
            # gndvi = torch.div(gndvi - (-0.7209), 0.5954 - (-0.7209))

            # RGB
            vari = torch.mul(g - r, g + r - b).unsqueeze(dim=0)
            # min: tensor(-0.4219)
            # max: tensor(0.1230)
            # MS
            # tensor(-0.3845)
            # tensor(0.5681)
            vari = torch.div(vari - (-0.4219), 0.1230 - (-0.4219))

            # RGB
            vdvi = torch.div(torch.mul(2, g - r - b), torch.mul(2, g + r + b)).unsqueeze(dim=0)
            # min: tensor(-0.5516)
            # max: tensor(0.3200)
            # MS
            # tensor(-0.6494)
            # tensor(0.2059)
            vdvi = torch.div(vdvi - (-0.5516), 0.3200 - (-0.5516))

            # Normalised Green-Blue Difference Vegetation Index
            ngbdi = torch.div((g - r), (g + b)).unsqueeze(dim=0)
            # RGB
            # min: tensor(-0.6831)
            # max: tensor(0.5952)
            # MS
            # tensor(-0.5145)
            # tensor(0.4930)
            ngbdi = torch.div(ngbdi - (-0.6831), 0.5952 - (-0.6831))

            # Normalised Green-Red Difference Vegetation Index
            ngrdi = torch.div((g - r), (g + r)).unsqueeze(dim=0)
            # RGB
            # min: tensor(-0.4125)
            # max: tensor(0.6296)
            # MS
            # tensor(-0.3101)
            # tensor(0.5066)
            ngrdi = torch.div(ngrdi - (-0.4125), 0.6296 - (-0.4125))

            t_im = torch.cat((t_im, vari, vdvi, ngbdi, ngrdi), dim=0)

        if self.target_transform is not None:
            t_im = self.target_transform(t_im)

        return t_im, img_path

    def __len__(self):
        return len(self.samples)
