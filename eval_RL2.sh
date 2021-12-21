path1=/home/liuhong/myworkspace/RL_exp/RL-std-only-ds/best_DS
path2=experiments_21/all_turn-level-us-12-1_sd11_lr0.0001_bs8_ga4/best_score_model
python session.py -mode test\
 -cfg DS_path=$path1 US_path=$path2\
 debugging=False\
 beam_search=False beam_size=10\
 DS_device=$1 US_device=$1