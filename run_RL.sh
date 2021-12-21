#path1=experiments_21/all_turn-level-ds-otl-12-6_sd11_lr0.0001_bs4_ga8/best_score_model
path1=experiments_21/all_turn-level-ds-S_sd11_lr0.0001_bs8_ga4/best_score_model
path2=experiments_21/all_turn-level-us-12-1_sd11_lr0.0001_bs8_ga4/best_score_model
#path2=RL_exp/RL-std-only-us/best_US
python session.py -mode train\
 -cfg use_scheduler=False\
 lr=2e-5\
 weight_decay_count=100\
 seed=11\
 epoch_num=100\
 init_eval=False\
 training_batch_size=16\
 rl_accumulation_steps=12\
 interaction_batch_size=32\
 DS_path=$path1 US_path=$path2\
 DS_device=$1 US_device=$1\
 rl_for_bspn=True\
 non_neg_reward=True\
 clip_grad=True\
 loss_reg=True\
 rl_dial_per_epoch=128\
 joint_train_ds=False\
 joint_train_us=True\
 simple_reward=True\
 simple_training=False\
 exp_no=RL-old-ds-new-us-only-us\
 rl_iterate=True\
 RL_ablation=False\
 iterative_update=False\
 notes=''