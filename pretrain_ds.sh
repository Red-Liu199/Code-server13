ratio=$2
posterior=False

if [ $posterior == True ]
then
    exp_no=DS_pos_${ratio}
    save_type='min_loss'
else
    exp_no=DS_${ratio}
    save_type='max_score'
fi
python main_turn_level.py -mode pretrain\
    -cfg  lr=1e-4\
    gradient_accumulation_steps=4 batch_size=8\
    epoch_num=50\
    cuda_device=$1\
    train_us=False\
    spv_proportion=$ratio\
    posterior_train=$posterior\
    save_type=$save_type\
    exp_no=$exp_no