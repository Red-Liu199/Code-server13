posterior=False
python main_turn_level_us.py -mode train\
    -cfg  lr=1e-4\
    gradient_accumulation_steps=4 batch_size=8\
    epoch_num=50\
    cuda_device=$1\
    train_us=True\
    posterior_train=$posterior\
    save_type=max_score\
    user_nlu=False\
    only_target_loss=True\
    exp_no=US-without-NLU-otl