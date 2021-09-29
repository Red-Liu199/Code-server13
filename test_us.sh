path=../distilgpt2
python pretrain.py -mode test\
    -cfg gpt_path=$path  cuda_device=$1\
    model_act=True dataset=1\
    debugging=True\
    fix_data=False\
    train_us=True