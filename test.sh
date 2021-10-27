path=experiments_21/turn-level-DS/best_score_model
#path=RL_exp/rl-10-19-use-scheduler/best_DS
python train_semi.py -mode test\
    -cfg gpt_path=$path  cuda_device=$1\
    fast_validate=True model_act=True dataset=1\
    debugging=False\
    fix_data=True\
    turn_level=True\
    input_history=False\
    input_prev_resp=True\
    test_unseen_act=True