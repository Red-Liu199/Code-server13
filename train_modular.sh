# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
# This script run the supervised baseline with 100% labeled data.
python pretrain.py -mode train\
    -cfg  lr=1e-4\
    gradient_accumulation_steps=4 batch_size=8\
    epoch_num=40\
    exp_no=DST-1-5\
    cuda_device=$1\
    only_target_loss=True\
    loss_reg=True\
    train_modular=True\
    modular=dst