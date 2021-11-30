path1=RL_exp/rl-large_batch-iterate/best_DS
path2=experiments_21/all_turn-level-us-111_sd11_lr0.0001_bs8_ga4/best_score_model

python session.py -mode test\
 -cfg DS_path=$path1 US_path=$path2\
 debugging=False\
 beam_search=False beam_size=10\
 DS_device=$1 US_device=$1

path1=experiments_21/all_turn-level-ds-S_sd11_lr0.0001_bs8_ga4/best_score_model
path2=RL_exp/rl-large_batch-iterate/best_US

python session.py -mode test\
 -cfg DS_path=$path1 US_path=$path2\
 debugging=False\
 beam_search=False beam_size=10\
 DS_device=$1 US_device=$1