path=RL_exp/rl-10-11/best_US
python pretrain.py -mode test\
    -cfg gpt_path=$path  cuda_device=$1\
    model_act=True dataset=1\
    eval_batch_size=32\
    train_us=True