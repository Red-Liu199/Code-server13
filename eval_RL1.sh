path1=RL_exp/RL-simple_reward-only-US/best_DS
path2=RL_exp/RL-simple_reward-only-US/best_US

python session.py -mode test\
 -cfg DS_path=$path1 US_path=$path2\
 debugging=False\
 beam_search=False beam_size=10\
 DS_device=$1 US_device=$1

path1=RL_exp/rl-large_batch-iterate/best_DS

python session.py -mode test\
 -cfg DS_path=$path1 US_path=$path2\
 debugging=False\
 beam_search=False beam_size=10\
 DS_device=$1 US_device=$1