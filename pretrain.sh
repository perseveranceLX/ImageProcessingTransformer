CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --seed 0 \
--lr 7e-5 \
--save-path "./ckpt" \
--epochs 300 \
--data /data1/share/remote_sensing/patched_images/rgb_superview \
--batch-size 360 \
| tee ./logs/v2_bs360_lr7e-5_epoch300.log

# finetune option "denoise30", "denoise50", "SRx2", "SRx3", "SRx4", "dehaze"