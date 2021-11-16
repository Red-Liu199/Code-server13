# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
# This script run the supervised baseline with 100% labeled data.
python train_semi.py -mode train\
    -cfg  lr=1e-4\
    gradient_accumulation_steps=16 batch_size=2\
    epoch_num=50\
    exp_no=baseline2\
    cuda_device=$1\
    save_type=max_score\
    fix_data=False\
    same_eval_as_cambridge=True\
    turn_level=False\
    only_target_loss=True\
    input_history=False\
    input_prev_resp=True\
    seed=2