#path1=experiments_21/turn-level-DS-100/best_score_model
path1=experiments_21/turn-level-DS-97_34-otl/best_score_model
path2=experiments_21/US-with-NLU/best_score_model
exp_name=RL-3-10-iter
python session.py -mode train\
 -cfg use_scheduler=False\
 lr=2e-5\
 weight_decay_count=150\
 seed=11\
 epoch_num=150\
 training_batch_size=16\
 rl_accumulation_steps=12\
 interaction_batch_size=32\
 DS_path=$path1 US_path=$path2\
 DS_device=$1 US_device=$1\
 rl_for_bspn=True\
 non_neg_reward=True\
 rl_dial_per_epoch=128\
 joint_train_ds=True\
 joint_train_us=True\
 simple_reward=True\
 simple_training=False\
 exp_no=$exp_name\
 rl_iterate=True\
 RL_ablation=False\
 iterative_update=True\
 on_policy=True\
 beam_search=True\
 user_nlu=True
