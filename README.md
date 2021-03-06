# ImageProcessingTransformer
Third party Pytorch implement of Image Processing Transformer (Pre-Trained Image Processing Transformer arXiv:2012.00364v2)

The latest version contains some important modifications according to the official mindspore implementation. It makes convergecy a lot faster. Please make sure you update to the latest version.

only contain model definition file and train/test file. Dataloader file is not yet released. You could implement your own dataloader. It may be released in the next version.

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
