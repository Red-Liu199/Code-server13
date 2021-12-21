# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
# This script run the supervised baseline with 100% labeled data.
ratio=$2
python pretrain.py -mode pretrain\
    -cfg  lr=1e-4\
    gradient_accumulation_steps=16 batch_size=2\
    seed=11\
    epoch_num=80\
    spv_proportion=$ratio\
    exp_no=UBAR-${ratio}\
    cuda_device=$1\
    save_type=max_score\
    turn_level=False\
    only_target_loss=True\
    loss_reg=True