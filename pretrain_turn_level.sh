# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
# This script run the supervised baseline with 100% labeled data.
python pretrain.py -mode train\
    -cfg  lr=1e-4\
    gradient_accumulation_steps=2 batch_size=16\
    seed=11\
    epoch_num=50\
    exp_no=test\
    cuda_device=$1\
    train_us=False\
    save_type=max_score\
    turn_level=True\
    only_target_loss=True\
    input_history=False\
    input_prev_resp=True\
    loss_reg=True