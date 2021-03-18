import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, dataloader
import os
import glob
import time
import cv2
from imagecorruptions import corrupt

from torchvision import transforms

def augment(l, hflip=True, vflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = vflip and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        return img

    return [_augment(_l) for _l in l]

def add_gaussian_noise(src_img, sigma):
    noise = np.random.normal(scale=sigma, size=src_img.shape)
    noise = noise.round()
    img_noisy = src_img.astype(np.int16) + noise.astype(np.int16)
    img_noisy = img_noisy.clip(0, 255).astype(np.uint8)
    return img_noisy

def add_fog(src_img, severity=1):
    """
        Available severity value are Integer [1, 5].
    """
    foggy = corrupt(src_img, corruption_name="fog", severity=severity)
    return foggy

def multiple_file_types(root_dir, types=[]):
    file_list = []
    for img_type in types:
        file_list += glob.glob(os.path.join(root_dir, img_type))
    return file_list



class DenoiseDataset(Dataset):
    def __init__(self, root_dir, sigma, transform=None):
        super(DenoiseDataset).__init__()
        self.file_list = self._make_dataset(root_dir)
        self.sigma = sigma
        self.transform = transform

    def _make_dataset(self, root_dir):
        file_list = multiple_file_types(root_dir, ["*.png", "*.jpg", "*.jpeg", "*.tif"])
        file_list.sort()
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        src_img = cv2.imread(self.file_list[index])
        trg_img = add_gaussian_noise(src_img, sigma=self.sigma)
        return self.transform(src_img), self.transform(trg_img)

class DehazeDataset(Dataset):
    def __init__(self, root_dir, severity, transform=None):
        super(DehazeDataset).__init__()
        self.file_list = self._make_dataset(root_dir)
        self.severity = severity
        self.transform = transform

    def _make_dataset(self, root_dir):
        file_list = multiple_file_types(root_dir, ["*.png", "*.jpg", "*.jpeg", "*.tif"])
        file_list.sort()
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        src_img = cv2.imread(self.file_list[index])
        trg_img = add_fog(src_img, severity=self.severity)
        return self.transform(src_img), self.transform(trg_img)

class SRDataset(Dataset):
    def __init__(self, root_dir, scale, mode='bilinear', transform=None):
        super(SRDataset).__init__()
        self.file_list = self._make_dataset(root_dir)
        self.transform = transform
        
        if scale in ["x2", "X2"]:
            self.scale = 0.5
        elif scale in ["x3", "X3"]:
            self.scale = 0.3333
        elif scale in ["x4", "X4"]:
            self.scale = 0.25
        else:
            raise NotImplementedError

        if mode == "bilinear":
            self.mode = cv2.INTER_LINEAR
        elif mode == "bicubic":
            self.mode = cv2.INTER_CUBIC
        else:
            raise NotImplementedError

    def _make_dataset(self, root_dir):
        file_list = multiple_file_types(root_dir, ["*.png", "*.jpg", "*.jpeg", "*.tif"])
        file_list.sort()
        return file_list   

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        src_img = cv2.imread(self.file_list[index])
        hr_h, hr_w = src_img.shape[:2]
        lr_h = round(hr_h * self.scale)
        lr_w = round(hr_w * self.scale)
        trg_img = cv2.resize(src_img, (lr_h, lr_w), interpolation=self.mode)
        return self.transform(src_img), self.transform(trg_img)


class ImageProcessDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super(ImageProcessDataset).__init__()
        self.file_list = self._make_dataset(root_dir)
        self.transform = transform

    def _make_dataset(self, root_dir):
        file_list = multiple_file_types(root_dir, ["*.png", "*.jpg", "*.jpeg", "*.tif"])
        file_list.sort()
        return file_list

    def __len__(self):
        return len(self.file_list)

    def _scale_size(self, src_h, src_w, scale):
        return (round(src_h / scale), round(src_w / scale))

    def __getitem__(self, index):
        src_img = cv2.imread(self.file_list[index])
        src_h, src_w, _ = src_img.shape
        target_group = []
        target_group.append(add_gaussian_noise(src_img, sigma=30))
        target_group.append(add_gaussian_noise(src_img, sigma=50))
        target_group.append(add_fog(src_img, severity=1))
        target_group.append(cv2.resize(src_img, self._scale_size(src_h, src_w, 2), cv2.INTER_LINEAR))
        target_group.append(cv2.resize(src_img, self._scale_size(src_h, src_w, 3), cv2.INTER_LINEAR))
        target_group.append(cv2.resize(src_img, self._scale_size(src_h, src_w, 4), cv2.INTER_LINEAR))

        if self.transform:
            return self.transform(src_img), [self.transform(tgt) for tgt in target_group]
        else:
            return src_img, target_group


class ImageProcessingIter(object):
    def __init__(self, datasets=[], batch_size=1, shuffle=False, 
                       sampler=None, num_workers=0):
        self.dataloaders = []
        for dataset in datasets:
            dataloader = DataLoader(dataset, 
                                    batch_size=batch_size, 
                                    shuffle=shuffle, 
                                    sampler=sampler,
                                    num_workers=num_workers)
            self.dataloaders.append(dataloader)

        self.iters = []
        self.num = 0
        self.stop_num = len(self.dataloaders[0])
        self._reset()

    def __iter__(self):
        return self

    def _reset(self):
        self.num = 0
        self.iters.clear()
        for dataloader in self.dataloaders:
            self.iters.append(iter(dataloader))

    def __next__(self):
        task_id = random.randint(0, 5)
        self.num += 1
        src, trg = self.iters[task_id].next()
        if self.num > self.stop_num:
            self._reset()
            raise StopIteration

        return src, trg, task_id





if __name__ == "__main__":
    root_dir = "./data/train/"

    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                                ])
    dataset = ImageProcessDataset(root_dir, transform=trans)
    train_loader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=16)

    start = time.time()
    count = 0
    for i in range(5):
        for step, (src, tgt_group) in enumerate(train_loader):
            task_id = random.randint(0, 5)   # [0,...,5]
            src, taregt, task_id = src, tgt_group[task_id], task_id

            
            print("elapse time: {}s".format(time.time() - start))

            start = time.time()
            count += 1

    print("total iter: ", count)
 

