path=experiments_21/baseline_fix_1/best_score_model
python train_semi.py -mode test\
    -cfg gpt_path=$path  cuda_device=$1\
    fast_validate=True model_act=True dataset=1\
    debugging=False\
    col_samples=True\
    fix_data=True\
    turn_level=False\
    input_history=False\
    input_prev_resp=True