# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
# This script run the supervised baseline with 100% labeled data.
python pretrain_test.py -mode train\
    -cfg  lr=1e-4\
    gradient_accumulation_steps=8 batch_size=4\
    epoch_num=50\
    exp_no=turn-level-dst\
    cuda_device=$1\
    only_target_loss=False\
    loss_reg=True\
    modular=dst