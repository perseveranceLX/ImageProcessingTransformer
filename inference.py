import argparse
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from ipt import ipt_base


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
    tid2ps = [48, 48, 24, 16, 12, 48] # task_id to patchsize
    inputs = [torch.ones(1, 3, p, p) for p in tid2ps]
    task_ids = [i for i in range(6)]
    for input, task_id in zip(inputs, task_ids):
        model.set_task(task_id)
        output = model(input)
        print(output.shape)
    
if __name__ == '__main__':
    main()
    