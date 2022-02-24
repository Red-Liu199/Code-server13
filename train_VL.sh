ratio=$2
posterior=False
python main_turn_level.py -mode pretrain\
    -cfg  lr=1e-4\
    seed=11\
    gradient_accumulation_steps=4 batch_size=8\
    epoch_num=50\
    cuda_device=0\
    spv_proportion=$ratio\
    posterior_train=$posterior\
    save_type=max_score\
    turn_level=True\
    input_history=False\
    input_prev_resp=True\
    exp_no=turn_level_${ratio}
path1=experiments_21/all_turn_level_${ratio}_sd11_lr0.0001_bs8_ga4/best_score_model
path2=experiments_21/all_turn_level_${ratio}_pos_sd11_lr0.0001_bs8_ga4/best_loss_model

python main_turn_level.py -mode semi_VL\
    -cfg PrioriModel_path=$path1 PosteriorModel_path=$path2\
    lr=1e-4\
    gradient_accumulation_steps=4 batch_size=8\
    epoch_num=40\
    cuda_device=$1\
    spv_proportion=$ratio\
    exp_no=VL-turn-level-gen-domain-${ratio}\
    turn_level=True\
    debugging=False\
    use_true_domain_for_ctr_train=True
