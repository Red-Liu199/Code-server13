path=/home/liuhong/myworkspace/RL_exp/rl-new-us/best_US
python pretrain.py -mode test\
    -cfg gpt_path=$path  cuda_device=$1\
    model_act=True dataset=1\
    eval_batch_size=32\
    train_us=True