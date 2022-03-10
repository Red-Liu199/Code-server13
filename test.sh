path=/home/liuhong/myworkspace/RL_exp/RL-3-9-joint-nlu/last_epoch_DS
python main_turn_level.py -mode test\
    -cfg gpt_path=$path  cuda_device=$1\
    fast_validate=True model_act=True dataset=1\
    turn_level=True\
    input_history=False\
    input_prev_resp=True\
    eval_batch_size=32\
    use_existing_result=True\
    save_prob=False
