
ratio=10
posterior=True
python train_semi.py -mode pretrain\
    -cfg  lr=1e-4\
    gradient_accumulation_steps=16 batch_size=2\
    epoch_num=50\
    cuda_device=1\
    spv_proportion=$ratio\
    model_act=True\
    posterior_train=$posterior\
    save_type=min_loss