# ImageProcessingTransformer
Third party Pytorch implement of Image Processing Transformer (Pre-Trained Image Processing Transformer arXiv:2012.00364v2)

only contain model definition file and train/test file. Dataloader file if not yet released. 
To pretrain on random task

    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --seed 0 \
    --lr 4e-5 \
    --save-path "./ckpt" \
    --epochs 30 \
    --reset-epoch \
    --data /data1/share/remote_sensing/patched_images/rgb_superview \
    --batch-size 180 \
    --resume /home/lizhexin/deit/ckpt/dehaze/checkpoint.pth.tar \
    --task "dehaze" \
    | tee ./logs/dehaze_bs320_lr2e-5_30epoch.log
