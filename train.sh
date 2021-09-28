# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
# This script run the supervised baseline with 100% labeled data.
python train_semi.py -mode train\
    -cfg  lr=1e-4\
    gradient_accumulation_steps=8 batch_size=4\
    epoch_num=50\
    exp_no=turn-level-DS\
    cuda_device=$1\
    save_type=max_score\
    fix_data=True\
    turn_level=True\
    only_target_loss=False\
    input_history=False\
    input_prev_resp=True