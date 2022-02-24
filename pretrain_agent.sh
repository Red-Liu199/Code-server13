python pretrain.py -mode train\
    -cfg  lr=1e-4\
    gradient_accumulation_steps=16 batch_size=2\
    seed=123\
    epoch_num=50\
    exp_no=UBAR\
    cuda_device=$1\
    train_us=False\
    save_type=max_score\
    turn_level=False\
    only_target_loss=True\
    input_history=True\
    input_prev_resp=True\
    loss_reg=True