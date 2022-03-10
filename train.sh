# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
# This script run the supervised baseline with 100% labeled data.
python train_semi.py -mode train\
    -cfg  lr=1e-4\
    gradient_accumulation_steps=8 batch_size=4\
    epoch_num=50\
    exp_no=HRU\
    cuda_device=$1\
    save_type=max_score\
    turn_level=True\
    only_target_loss=True\
    input_history=True\
    input_prev_resp=True\
    seed=6