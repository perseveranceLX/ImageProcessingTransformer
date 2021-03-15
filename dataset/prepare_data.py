import random
import numpy as np
import torch.utils.data as data
import os
import glob
import cv2
from imagecorruptions import corrupt
from tqdm import tqdm

from torchvision import transforms

import pdb

def prepare_patches(image, patch_size):
    scale = image.shape[0] // patch_size
    patches = []
    for h_split in np.split(image, scale, axis=0):   ## split img to rows 
        for v_split in np.split(h_split, scale, axis=1):  ## split a row to colume
            patches.append(v_split)

    return patches

def multiple_file_types(root_dir, types=[]):
    file_list = []
    for img_type in types:
        path = os.path.join(root_dir, img_type)
        file_list += glob.glob(path)
    return file_list



if __name__ == "__main__":
    data_dir = "/home/lizhexin/dataset/remote_sense"
    target_dir = "/home/lizhexin/dataset/patched_images"

    trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                # transforms.Resize(512),
                transforms.RandomCrop((480, 480)),
                transforms.ToTensor()])

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # pdb.set_trace()

    print(len(multiple_file_types(data_dir, ["*/*.jpg", "*/*.png", "*/*.tif"])))
    #print(len(multiple_file_types(data_dir, ["*.jpg", "*.png", "*.tif"])))

    for path in tqdm(multiple_file_types(data_dir, ["*/*.jpg", "*/*.png", "*/*.tif"])):
        img = cv2.imread(path)
        img = trans(img).numpy() * 255.
        img = np.transpose(img, (1, 2, 0))
        file_name = os.path.basename(path)
        patches = prepare_patches(image=img, patch_size=48)
        for i in range(len(patches)):
            sub_dir = path.split('/')[-2]
            patch_name = "{}_{}_{}.png".format(sub_dir, file_name.split('.')[0], i)

            cv2.imwrite(os.path.join(target_dir, patch_name), patches[i])

        break

        
        
