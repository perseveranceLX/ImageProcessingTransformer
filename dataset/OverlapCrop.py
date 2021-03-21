import numpy as np
import cv2
import math
import pdb
from tqdm import tqdm
import torch
import random

class OverlapCrop():
    def __init__(self, img=None, patch_size = 48, overlap=10):
        self.img = img
        self.patches = []
        self.patch_size = patch_size
        self.overlap = overlap

    def _set_param(self):
        stride = self.patch_size - self.overlap
        h, w = self.img.shape[:2]
        N_h = math.ceil((h - self.patch_size) / stride) + 1
        N_w = math.ceil((w - self.patch_size) / stride) + 1
        padding = (N_h - 1) * stride + self.patch_size - h
        padding_h_1 = padding // 2
        padding_h_2 = padding - padding_h_1
        padding = (N_w - 1) * stride + self.patch_size - w
        padding_w_1 = padding // 2
        padding_w_2 = padding - padding_w_1
        self.padding = ((padding_h_1, padding_h_2), (padding_w_1, padding_w_2), (0, 0))
        self.N_h = N_h
        self.N_w = N_w
        self.pad_img = np.pad(self.img, self.padding, mode='reflect')

        # print("padding image shape: ", self.pad_img.shape)

    def _vcrop(self, array, num, size=48, overlap=10):
        patches = []
        patches.append(array[:size, :, :])
        pointer = size - overlap
        for i in range(1, num):
            patches.append(array[pointer:pointer + size, :, :])
            pointer = pointer + size - overlap

        return patches
    
    def _hcrop(self, array, num, size=48, overlap=10):
        patches = []
        patches.append(array[:, :size, :])
        pointer = size - overlap
        for i in range(1, num):
            patches.append(array[:, pointer:pointer + size, :])
            pointer = pointer + size - overlap

        return patches

    def _hpad(self, arr_l, arr_r, overlap=10):
        if arr_l.size == 1:
            return arr_r.astype(np.float32)
        h, w, c = arr_l.shape
        patch_w = arr_r.shape[1]
        if overlap == 10:
            weight = np.array([0.03, 0.08, 0.15, 0.25, 0.5, 0.5, 0.75, 0.85, 0.92, 0.97])
            weight = np.expand_dims(weight, axis=0)
            weight = np.expand_dims(weight, axis=2)
        result = np.zeros((h, w + patch_w - overlap, c))
        result[:, :w-overlap, :] = arr_l[:, :w-overlap, :]
        tmp = arr_l[:, -overlap:, :] * (1-weight) + arr_r[:, :overlap, :] * weight
        # tmp = arr_l[:, -overlap:, :]
        # tmp = arr_r[:, :overlap, :]
        result[:, w-overlap:w, :] = tmp
        result[:, w:, :] = arr_r[:, overlap:, :]
        return result

    def _vpad(self, arr_up, arr_down, overlap=10):
        if arr_up.size == 1:
            return arr_down.astype(np.float32)
        h, w, c = arr_up.shape
        pathch_h = arr_down.shape[0]
        if overlap == 10:
            weight = np.array([0.03, 0.08, 0.15, 0.25, 0.5, 0.5, 0.75, 0.85, 0.92, 0.97])
            weight = np.expand_dims(weight, axis=1)
            weight = np.expand_dims(weight, axis=2)
        result = np.zeros((h + pathch_h - overlap, w, c))
        result[:h-overlap, :, :] = arr_up[:h-overlap, :, :]
        tmp = arr_up[-overlap:, :, :] * (1-weight) + arr_down[:overlap, :, :] * weight
        # tmp = arr_up[-overlap:, :, :]
        # tmp = arr_down[:overlap, :, :]
        result[h-overlap:h, :, :] = tmp
        result[h:, :, :] = arr_down[overlap:, :, :]
        return result

    def fold(self):
        self._set_param()
        img = np.array(0)
        for i in range(self.N_h):
            v_patch = np.array(0)
            for j in range(self.N_w):
                patch = self.patches[i*self.N_w+j]
                v_patch = self._hpad(v_patch, patch, self.overlap)
            
            img = self._vpad(img, v_patch, self.overlap)

        (pad_h, _), (pad_w, _), _= self.padding
        h, w = self.img.shape[:2]
        self.img = img[pad_h:pad_h+h, pad_w:pad_w+w, :]
        return self.img

    def unfold(self):
        self._set_param()
        patches = []
        for v_split in self._vcrop(self.pad_img, self.N_h, self.patch_size, self.overlap):
            for h_split in self._hcrop(v_split, self.N_w, self.patch_size, self.overlap):
                patches.append(h_split)
        self.patches = patches
        return self.patches

    def get_patches(self):
        return self.patches

    def get_image(self):
        return self.img

    def set_patches(self, patches):
        self.patches = patches

    def set_image(self, img):
        self.img = img




if __name__ == "__main__":
    # img_path = "../test/female_281.jpg"
    img_path = "../results/0_0_in.png"

    img = cv2.imread(img_path)
    input = OverlapCrop(img, patch_size=48, overlap=10)
    patches = input.unfold()

    
    out_img = np.zeros(img.shape)
    output = OverlapCrop(out_img, patch_size=48, overlap=10)
    output.set_patches(patches)
    output.fold()
    out_img = output.get_image()
    print(out_img[:5, :5, :1])

    cv2.imwrite("./test.png", out_img.astype(np.uint8))
    
