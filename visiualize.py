import argparse
import os
import random
import time
import warnings
from datetime import datetime
from collections import OrderedDict
import math
import random
import cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from ipt import ipt_base
from dataset.dataset import *


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-i','--img', type=str, metavar='DIR', default='',
                    help='path to image')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to checkpoint (default: none)')


def main():
    args = parser.parse_args()

    print("=> creating model '{}'".format("ipt_base"))
    model = ipt_base()

    # DataParallel will divide and allocate batch_size to all available GPUs
    model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            #args.start_epoch = checkpoint['epoch']
            args.start_epoch = 6
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    path = args.img
    img = cv2.imread(path)
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                                ])
    input = trans(img).unsqueeze(0).permute(0, 3, 1, 2) # input now is a tensor of size (1, 3, 512, 512)

    def patching(x):
        N, C, H, W = ori_shape = x.shape
        
        p = 48
        num_patches = (H // p) * (W // p)
        out = torch.zeros((N, num_patches, self.dim)).cuda()
        #print(f"feature map size: {ori_shape}, embedding size: {out.shape}")
        i, j = 0, 0
        for k in range(num_patches):
            if i + p > W:
                i = 0
                j += p
            out[:, k, :] = x[:, :, i:i+p, j:j+p].flatten(1)
            i += p
        return out, ori_shape

    def __init__(self, patch_size=1, in_channels=64):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = None
        self.dim = self.patch_size ** 2 * in_channels


    def forward(self, x, ori_shape):
        N, num_patches, dim = x.shape
        _, C, H, W = ori_shape
        p = self.patch_size
        out = torch.zeros(ori_shape).cuda()
        i, j = 0, 0
        for k in range(num_patches):
            if i + p > W:
                i = 0
                j += p
            out[:, :, i:i+p, j:j+p] = x[:, k, :].reshape(N, C, p, p)
            #out[:, k, :] = x[:, :, i:i+p, j:j+p].flatten(1)
            i += p
        return out


def train(train_loader, model, criterion, optimizer, epoch, args):
    #train for one epoch
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    if args.lr_policy == 'step':
        local_lr = adjust_learning_rate(optimizer, epoch, args)
    elif args.lr_policy == 'epoch_poly':
        local_lr = adjust_learning_rate_epoch_poly(optimizer, epoch, args)
        

    for i, (target, input, task_id) in enumerate(train_loader):
        # set random task
        model.module.set_task(task_id)
        #print(f"Iter {i}, task_id: {task_id}")
        #for m in model.module.modules():
           # if isinstance(m, )
            #print(m.weight.device)
        global_iter = epoch * args.epoch_size + i
        
        if args.lr_policy == 'iter_poly':
            local_lr = adjust_learning_rate_poly(optimizer, global_iter, args)
        elif args.lr_policy == 'cosine':
            local_lr = adjust_learning_rate_cosine(optimizer, global_iter, args)
        
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)

        #print(output.device, target.device)
        loss = criterion(output, target.cuda())

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/random]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'LR: {lr: .6f}'.format(
                   epoch, i, batch_time=batch_time,
                   data_time=data_time, loss=losses, lr=local_lr))

if __name__ == '__main__':
    main()
