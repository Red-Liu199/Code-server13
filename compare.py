import json
path1='experiments_21/DS_base/best_score_model/unsuccess_dials.json'
path2='RL_exp/rl_joint_no_scheduler_neg_reward/best_DS/unsuccess_dials.json'
data0=json.load(open(path1,'r', encoding='utf-8'))
data1=json.load(open(path2,'r', encoding='utf-8'))
for dial_id in data0:
    if dial_id not in data1:
        print('unsuccess in baseline but success after RL',dial_id)
for dial_id in data1:
    if dial_id not in data0:
        print('success in baseline but unsuccess after RL', dial_id)