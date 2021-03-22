# ImageProcessingTransformer
Third party Pytorch implement of Image Processing Transformer (Pre-Trained Image Processing Transformer arXiv:2012.00364v2)

only contain model definition file and train/test file. Dataloader file if not yet released. 

To pretrain on random task

    python main.py --seed 0 \
    --lr 5e-5 \
    --save-path "./ckpt" \
    --epochs 300 \
    --data path-to-data \
    --batch-size 256

To finetune on a specific task

    python main.py --seed 0 \
    --lr 2e-5 \
    --save-path "./ckpt" \
    --epochs 30 \
    --reset-epoch \
    --data path-to-data \
    --batch-size 256 \
    --resume path-to-pretrain-model \
    --task "dehaze"
    
To eval on a specific task

    python main.py --seed 0 \
    --eval-data path-to-val-data \
    --batch-size 256 \
    --eval \
    --resume path-to-model \
    --task "dehaze"
