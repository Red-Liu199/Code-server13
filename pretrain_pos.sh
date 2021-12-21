
ratio=$2
posterior=True
python main_turn_level.py -mode pretrain\
    -cfg  lr=1e-4\
    gradient_accumulation_steps=4 batch_size=8\
    epoch_num=50\
    cuda_device=$1\
    spv_proportion=$ratio\
    posterior_train=$posterior\
    save_type=min_loss\
    input_history=False\
    input_prev_resp=True\
    exp_no=turn_level_${ratio}_pos