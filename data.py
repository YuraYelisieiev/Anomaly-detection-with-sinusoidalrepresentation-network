from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
import os
import scipy
from dataio import get_mgrid
from copy import deepcopy
from utils import tri_mirror
import pytorch_lightning as pl
import albumentations as A


class RIVETSDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()])

        self.batch_size = self.config['data']['batch_size']

    def setup(self, stage_name):
        train_img_dataset = TextureDataset(self.config, self.transform, "train")
        val_img_dataset = TextureDataset(self.config, self.transform, "val")

        self.rivets_train = Implicit2DWrapper(train_img_dataset, sidelength=self.config['data']['after_crop_size'] // 2,
                                              compute_diff='all')
        self.rivets_val = Implicit2DWrapper(val_img_dataset, sidelength=self.config['data']['after_crop_size'] // 2,
                                            compute_diff='all')

    def train_dataloader(self):
        return DataLoader(self.rivets_train, batch_size=self.batch_size, shuffle=True, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.rivets_val, batch_size=self.batch_size, shuffle=False, num_workers=16)


class ProcessImage:
    def __init__(self, **kvargs):
        pass

    def __call__(self, img):
        return img


class Dummy(ProcessImage):
    def __call__(self, img):
        return img


def get_dict_value(d, key, default_value):
    if key in d.keys():
        return d[key]
    return default_value


class TextureDataset(Dataset):
    def __init__(self, config, transform, part):
        self.part = part

        if self.part == "train":
            self.root_dir = config['data']['train_dir']
        else:
            self.root_dir = config['data']['val_dir']

        self.image_size = config['data']['image_size']
        self.after_crop_size = config['data']['after_crop_size']
        self.center = self.after_crop_size // 2
        self.cv = self.after_crop_size // 4
        self.crop_start = self.center - self.cv
        self.crop_end = self.center + self.cv

        self.image_type = config['data']['image_type']
        self.transform = transform
        self.img_channels = 1

        self.rotate = transforms.Compose([
            transforms.RandomRotation(90)
        ])

        self.crop = transforms.Compose([
            transforms.CenterCrop(self.after_crop_size)
        ])

        self.color_jitter = transforms.ColorJitter(brightness=0.35,
                                                   contrast=0.35,
                                                   saturation=0.35,
                                                   hue=0.35)

        # self.elastic = A.ElasticTransform(p=0.5, alpha=self.after_crop_size * 2,
        #                                   sigma=self.after_crop_size * 0.15,
        #                                   alpha_affine=self.after_crop_size * 0.15)

        self.file_list = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                self.file_list.append(os.path.join(root, file))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        X = dict()
        img_path = self.file_list[idx]
        rivet = Image.open(img_path).resize((self.image_size, self.image_size))

        # if self.part == "train":
        #     # rivet = self.elastic(image=np.array(rivet))['image']
        #     rivet = Image.fromarray(rivet)

        rotated = deepcopy(rivet)

        rivet = self.crop(rivet)
        rotated = self.rotate(rotated)
        rotated = self.crop(rotated)

        if self.image_type == "Corrupted":
            full_rivet = deepcopy(rivet)
            rivet = np.array(rivet)
            center_rivet = deepcopy(rivet[self.crop_start: self.crop_end, self.crop_start: self.crop_end])
            rivet[self.crop_start: self.crop_end,self.crop_start: self.crop_end] = 0.0

            rivet = Image.fromarray(rivet)
            center_rivet = center_rivet.astype(np.uint8)
            center_rivet = Image.fromarray(center_rivet)

        elif self.image_type == "TriMir":
            rivet = tri_mirror(rivet, self.center, self.cv)

        rivet = self.transform(rivet)
        full_rivet = self.transform(full_rivet)
        center_rivet = self.transform(center_rivet)
        rotated = self.transform(rotated)

        center_rivet = center_rivet.unsqueeze(0)

        X['Rivet'] = rivet
        X['Center'] = center_rivet
        X['Rotated'] = rotated
        X['Full'] = full_rivet
        return X


class Implicit2DWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None, compute_diff=None):
        if isinstance(sidelength, int):
            sidelength = (sidelength, sidelength)
        self.sidelength = sidelength

        self.compute_diff = compute_diff
        self.dataset = dataset
        self.mgrid = get_mgrid(sidelength)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        borders = item['Rivet']
        img = item['Center']
        rotated = item['Rotated']

        if self.compute_diff == 'gradients':
            img *= 1e1
            gradx = scipy.ndimage.sobel(img.numpy(), axis=1).squeeze(0)[..., None]
            grady = scipy.ndimage.sobel(img.numpy(), axis=2).squeeze(0)[..., None]
        elif self.compute_diff == 'laplacian':
            img *= 1e4
            laplace = scipy.ndimage.laplace(img.numpy()).squeeze(0)[..., None]
        elif self.compute_diff == 'all':
            gradx = scipy.ndimage.sobel(img.numpy(), axis=1).squeeze(0)[..., None]
            grady = scipy.ndimage.sobel(img.numpy(), axis=2).squeeze(0)[..., None]
            laplace = scipy.ndimage.laplace(img.numpy()).squeeze(0)[..., None]

        img = img.squeeze(0)
        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

        in_dict = {'idx': idx, 'coords': self.mgrid, "border": borders, 'rotated': rotated}
        gt_dict = {'img': img}

        if self.compute_diff == 'gradients':
            gradients = torch.cat((torch.from_numpy(gradx).reshape(-1, 1),
                                   torch.from_numpy(grady).reshape(-1, 1)),
                                  dim=-1)
            gt_dict.update({'gradients': gradients})

        elif self.compute_diff == 'laplacian':
            gt_dict.update({'laplace': torch.from_numpy(laplace).view(-1, 1)})

        elif self.compute_diff == 'all':
            gradients = torch.cat((torch.from_numpy(gradx).reshape(-1, 1),
                                   torch.from_numpy(grady).reshape(-1, 1)),
                                  dim=-1)
            gt_dict.update({'gradients': gradients})
            gt_dict.update({'laplace': torch.from_numpy(laplace).view(-1, 1)})

        return in_dict, gt_dict

    def get_item_small(self, idx):
        img = self.dataset[idx]
        spatial_img = img.clone()
        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

        gt_dict = {'img': img}

        return spatial_img, img, gt_dict
