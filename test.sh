#path=experiments_21/DS_base/best_score_model
#path=/home/liuhong/myworkspace/experiments_21/DS_base/best_score_model
path=/home/liuhong/myworkspace/experiments_21/all_turn-level-ds-S_sd11_lr0.0001_bs8_ga4/best_score_model
#path=RL_exp/rl-10-26/best_DS
python train_semi.py -mode test\
    -cfg gpt_path=$path  cuda_device=$1\
    fast_validate=True model_act=True dataset=1\
    debugging=False\
    fix_data=True\
    turn_level=True\
    input_history=False\
    input_prev_resp=True\
    test_unseen_act=False\
    eval_resp_prob=False\
    same_eval_as_cambridge=True\
    use_existing_result=True\
    eval_as_simpletod=False\
    eval_batch_size=32