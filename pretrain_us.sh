ratio=$2
posterior=$3

if [ $posterior == True ]
then
    exp_no=US_pos_${ratio}
    save_type='min_loss'
else
    exp_no=US_${ratio}
    save_type='max_score'
fi
python main_turn_level.py -mode pretrain\
    -cfg  lr=1e-4\
    gradient_accumulation_steps=4 batch_size=8\
    epoch_num=50\
    cuda_device=$1\
    train_us=True\
    spv_proportion=$ratio\
    posterior_train=$posterior\
    save_type=$save_type\
    exp_no=$exp_no