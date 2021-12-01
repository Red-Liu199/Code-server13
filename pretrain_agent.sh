# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
# This script run the supervised baseline with 100% labeled data.
python pretrain.py -mode train\
    -cfg  lr=1e-4\
    gradient_accumulation_steps=4 batch_size=8\
    epoch_num=40\
    exp_no=turn-level-us-12-1\
    cuda_device=$1\
    train_us=True\
    save_type=max_score\
    turn_level=True\
    only_target_loss=True\
    input_history=False\
    input_prev_resp=True\
    loss_reg=True