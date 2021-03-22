import argparse
import os
import torch
import numpy as np

from ipt import ipt_base
from dataset.OverlapCrop import OverlapCrop
import cv2


parser = argparse.ArgumentParser(description='PyTorch ImageNet Testing')
parser.add_argument('-i','--img', type=str, metavar='DIR', default='',
                    help='path to input image')
parser.add_argument('-o','--out', type=str, metavar='DIR', default='',
                    help='path to output image')
parser.add_argument('--task_id', type=int, default=1,
                    help='indicate task to execute, [0, ..., 5]')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to checkpoint (default: none)')

task_dict = {0: (48, 10),
             1: (48, 10),
             2: (24, 5),
             3: (16, 4),
             4: (12, 3),
             5: (48, 10)}

def test(model, img, task):
    assert 0 <= task < 6
    patch_size, overlap = task_dict[task]
    img = (img / 255. - 0.5) / 0.5      # Normalize

    cropper = OverlapCrop(img, overlap=overlap, patch_size=patch_size)
    src_patches = cropper.unfold()

    input_batch = []
    for patch in src_patches:
        input = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(dim=0)     # HWC -> NCHW
        input_batch.append(input)
    input_batch = torch.cat(input_batch, dim=0)
    print(input_batch.shape, task)
    input_batch = input_batch[:1, :, :, :].cuda()
    model.set_task(task)
    # model.module.set_task(task)
    out_batch = model(input_batch).detach().cpu()
    out_patches = []
    for out in out_batch:
        patch = np.array(out).transpose(1, 2, 0)    # CHW -> HWC
        out_patches.append(patch)

    tgt = np.zeros_like(img)
    if task in [2, 3, 4]:
        tgt = np.zeros(tgt.shape * task)
    folder = OverlapCrop(tgt, patch_size=48, overlap=10)
    folder.set_patches(out_patches)
    out_img = folder.fold()
    out_img = (out_img * 0.5 + 0.5) * 255.

    return out_img.astype(np.uint8)

def main():
    args = parser.parse_args()

    print("=> creating model '{}'".format("ipt_base"))
    model = ipt_base().cuda()

    # DataParallel will divide and allocate batch_size to all available GPUs
    # model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("=> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            checkpoint, epoch = checkpoint['state_dict'], checkpoint['epoch']
            checkpoint = {k[7:] : v for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint)
            
            print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, epoch))
            
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.checkpoint))

    model.eval()

    input_img = cv2.imread(args.img)
    out_img = test(model, input_img, args.task_id)
    cv2.imwrite(args.out, out_img)

    
if __name__ == '__main__':
    main()
    