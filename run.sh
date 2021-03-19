CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --seed 0 \
--lr 1e-4 \
--data /data1/share/remote_sensing/patched_images/rgb_superview \
--batch-size 320 \
--resume /home/lizhexin/deit/ckpt/03-17-12_29_37/checkpoint.pth.tar \
| tee ./logs/bs320_lr1e-4.log