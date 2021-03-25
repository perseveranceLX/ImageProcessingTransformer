python main.py --seed 0 \
--lr 2e-5 \
--save-path "./ckpt" \
--epochs 30 \
--data path-to-data \
--batch-size 256 \
--resume path-to-checkpoint \
--task "dehaze"
# finetune option "denoise30", "denoise50", "SRx2", "SRx3", "SRx4", "dehaze"