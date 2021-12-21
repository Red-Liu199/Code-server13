ratio=$2
path=experiments_21/all_turn_level_${ratio}_sd11_lr0.0001_bs8_ga4/best_score_model
python main_turn_level.py -mode semi_ST\
    -cfg gpt_path=$path lr=1e-4\
    gradient_accumulation_steps=4 batch_size=8\
    epoch_num=40\
    cuda_device=$1\
    spv_proportion=$ratio\
    exp_no=ST-${ratio}\
    turn_level=True\
    debugging=False\
    init_eval=False
