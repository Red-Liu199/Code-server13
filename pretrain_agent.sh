# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
# This script run the supervised baseline with 100% labeled data.
python pretrain.py -mode train\
    -cfg  lr=1e-4\
    gradient_accumulation_steps=2 batch_size=16\
    epoch_num=40\
    exp_no=sys-model\
    cuda_device=$1\
    train_us=False\
    save_type=max_score\
    turn_level=True\
    only_target_loss=False\
    input_history=False\
    input_prev_resp=True\
    debugging=False\
    train_sys=True