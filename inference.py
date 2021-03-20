import argparse
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader

from ipt import ipt_base
from dataset.dataset import *
from dataset.OverlapCrop import OverlapCrop

import cv2
from tqdm import tqdm
import pdb


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-i','--img', type=str, metavar='DIR', default='',
                    help='path to image')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to checkpoint (default: none)')
parser.add_argument('--dir', type=str, metavar='DIR', default='',
                    help='path to image')
parser.add_argument('--out_dir', type=str, metavar='DIR', default='',
                    help='path to output image')

def test(test_loader, model, device, result_dir):
    model = model.to(device)
    for i, (tgt, src_group) in tqdm(enumerate(test_loader)):
        tgt = tgt[0]
        tgt = np.array(tgt).transpose(1, 2, 0)  # CHW -> HWC
        tgt = (tgt * 0.5 + 0.5) * 255.
        path_tgt = os.path.join(result_dir, "{}.png".format(i))
        cv2.imwrite(path_tgt, tgt)
        tgt = tgt.astype(np.uint8)
        for task_id in range(6):
            src = src_group[task_id][0]
            # print(tgt.shape)
            src = np.array(src).transpose(1, 2, 0)   # CHW -> HWC
            
            if task_id in [0, 1, 5]:
                patch_size = 48
                overlap = 10
            elif task_id == 2:
                patch_size = 24
                overlap = 5
            elif task_id == 3:
                patch_size = 16
                overlap = 4
            elif task_id == 4:
                patch_size = 12
                overlap = 3

            cropper = OverlapCrop(src, overlap=overlap, patch_size=patch_size)
            src_patches = cropper.unfold()
            input_batch = []
            out_batch = []
            for patch in src_patches:
                input = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(dim=0)
                # print(input.shape)
                input_batch.append(input)

            # input_batch = torch.cat(input_batch, dim=0)
            # input_batch = input_batch.to(device)
            model.set_task(task_id)
            for input in input_batch:
                input = input.to(device)
                out_batch.append(model(input).detach().cpu())

            out_batch = torch.cat(out_batch, dim=0)
            out_patches = []
            for out in out_batch:
                patch = np.array(out).transpose(1, 2, 0)
                out_patches.append(patch)

            folder = OverlapCrop(tgt)
            folder.set_patches(out_patches)
            out_img = folder.fold()
            out_img = (out_img * 0.5 + 0.5) * 255.
            src = (src * 0.5 + 0.5) * 255.
            out_img = np.clip(out_img, 0, 255)
            out_img = out_img.astype(np.uint8)
            src = src.astype(np.uint8)
            
            path_in = os.path.join(result_dir, "{}_{}_in.png".format(i, task_id))
            path_out = os.path.join(result_dir, "{}_{}_out.png".format(i, task_id))
            cv2.imwrite(path_out, out_img)
            cv2.imwrite(path_in, src)

        # break

def main():
    args = parser.parse_args()

    print("=> creating model '{}'".format("ipt_base"))
    device = torch.device("cuda")
    model = ipt_base()

    # DataParallel will divide and allocate batch_size to all available GPUs
    #model = torch.nn.DataParallel(model).cuda()

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
    
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                                ])

    dataset = ImageProcessDataset(args.dir, transform=trans)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    test(test_loader, model, device, result_dir=args.out_dir)
    # tid2ps = [48, 48, 24, 16, 12, 48] # task_id to patchsize
    # inputs = [torch.ones(1, 3, p, p) for p in tid2ps]
    # task_ids = [i for i in range(6)]
    # for input, task_id in zip(inputs, task_ids):
    #     model.set_task(task_id)
    #     output = model(input)
    #     print(output.shape)
    
if __name__ == '__main__':
    main()
    