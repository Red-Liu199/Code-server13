#path1=experiments_21/all_turn-level-ds-otl-12-6_sd11_lr0.0001_bs4_ga8/best_score_model
#path1=experiments_21/turn-level-DS/best_score_model
#path2=experiments_21/all_turn-level-us-12-1_sd11_lr0.0001_bs8_ga4/best_score_model
path1=/home/liuhong/myworkspace/RL_exp/RL-1-10-beam-1/best_DS
path2=/home/liuhong/myworkspace/RL_exp/RL-1-10-beam-1/best_US
python session.py -mode test\
 -cfg DS_path=$path1 US_path=$path2\
 debugging=False\
 beam_search=False beam_size=10\
 DS_device=$1 US_device=$1\
 fix_DST=False\
 DST_path=experiments_21/all_DST-1-5_sd11_lr0.0001_bs8_ga4/best_score_model