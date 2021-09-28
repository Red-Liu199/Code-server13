from config import global_config as cfg
from eval import MultiWozEvaluator
from reader import MultiWozReader
from transformers import GPT2Tokenizer
import json
import ontology

tokenizer=GPT2Tokenizer.from_pretrained('best_model')
reader = MultiWozReader(tokenizer)
evaluator = MultiWozEvaluator(reader)
#path='RL_exp/rl_fix_act/best_DS/online_result.json'
path='experiments_21/DS_base/best_score_model/online_mismatch.json'
data=json.load(open(path,'r', encoding='utf-8'))
#print('Total unsuccess dialogs:', len(data))
unsuccess_dict={'restaurant':0, 'hotel':0, 'attraction':0, 'train':0, 'taxi':0, 'police':0, 'hospital':0}
mismatch_dict={'restaurant':0, 'hotel':0, 'attraction':0, 'train':0, 'taxi':0, 'police':0, 'hospital':0}
mismatch_num=0
match_dials=[]
for dial_id in data:
    #print('Dial id:', dial_id)
    dial = data[dial_id]
    reqs = {}
    goal = {}
    counts = {}
    for req in evaluator.requestables:
        counts[req+'_total'] = 0
        counts[req+'_offer'] = 0
    if '.json' not in dial_id and '.json' in list(evaluator.all_data.keys())[0]:
        dial_id = dial_id + '.json'
    for domain in ontology.all_domains:
        if evaluator.all_data[dial_id]['goal'].get(domain):
            true_goal = evaluator.all_data[dial_id]['goal']
            goal = evaluator._parseGoal(goal, true_goal, domain)
    #print('Goal:\n', goal)
    for domain in goal.keys():
        reqs[domain] = goal[domain]['requestable']
    #print('Reqt:\n', reqs)
    #print('Dial:\n', dial)
    success, match, _, _ = evaluator._evaluateGeneratedDialogue(dial, goal, reqs, counts, soft_acc=True)
    #print('Success:{}, match:{}'.format(success, match))
    if len(match)<len(reqs):
        mismatch_num+=1
    else:
        match_dials.append(dial_id)
        #print(dial_id)
        #print('Reqt:\n', reqs)
        #print('Success:\n', success)

    for domain in reqs.keys():
        if domain not in match:
            mismatch_dict[domain]+=1
        if domain not in success:
            unsuccess_dict[domain]+=1
print('Total mismatch dialogs:', mismatch_num)
print('Mismatch:\n', mismatch_dict)
print('Unsuccess:\n', unsuccess_dict)
#print('Unsuccess but match dialogs:', match_dials)
        