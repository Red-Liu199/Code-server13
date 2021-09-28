
device=$1
ratio=10

path=experiments_21/all_BRU-10_sd11_lr0.0001_bs4_ga8/best_score_model
python session4RL.py\
    -cfg batch_size=4 gradient_accumulation_steps=8\
    lr=1e-6\
    DS_path=$path\
    epoch_num=40\
    rl_with_us=False\
    use_scheduler=False\
    traverse_data=False\
    rl_dial_per_epoch=1000\
    turn_level=True\
    input_history=False\
    input_prev_resp=True\
    exp_no=RL-offline-10\
    cuda_device=$device\
    init_eval=False
