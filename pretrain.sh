
ratio=$2
posterior=False
python main_turn_level.py -mode pretrain\
    -cfg  lr=1e-4\
    seed=11\
    gradient_accumulation_steps=4 batch_size=8\
    epoch_num=50\
    cuda_device=$1\
    spv_proportion=$ratio\
    posterior_train=$posterior\
    save_type=max_score\
    turn_level=True\
    input_history=True\
    input_prev_resp=True\
    exp_no=turn_level_HRU_${ratio}