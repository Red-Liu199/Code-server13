path=/home/liuhong/myworkspace/RL_exp/rl-10-12-neg/best_DS
python train_semi.py -mode test\
    -cfg gpt_path=$path  cuda_device=$1\
    fast_validate=True model_act=True dataset=1\
    debugging=False\
    fix_data=True\
    turn_level=True\
    input_history=False\
    input_prev_resp=True