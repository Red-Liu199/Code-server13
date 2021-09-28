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
save_path='experiments_21/DS_base/best_score_model/process_mismatch.json'
data=json.load(open(path,'r', encoding='utf-8'))
useless_key=['sys_intent', 'usr_intent', 'resp_gen', 'bspn_gen', 'dial_id']
new_data={}
print("Original mismatch dialogs:", len(data))
count=0
for dial_id, dial in data.items():
    reqs = {}
    goal = {}
    counts = {}
    for req in evaluator.requestables:
        counts[req+'_total'] = 0
        counts[req+'_offer'] = 0
    for domain in ontology.all_domains:
        if evaluator.all_data[dial_id]['goal'].get(domain):
            true_goal = evaluator.all_data[dial_id]['goal']
            goal = evaluator._parseGoal(goal, true_goal, domain)
    for domain in goal.keys():
        reqs[domain] = goal[domain]['requestable']
    success, match, _, _ = evaluator._evaluateGeneratedDialogue(dial, goal, reqs, counts, soft_acc=True)
    if len(match)<len(reqs):
        count+=1
    else:
        continue
    for turn in dial:
        for key in useless_key:
            turn.pop(key)
        turn['turn_domain']=' '.join(turn['turn_domain'])
    new_data[dial_id]={'goal':goal, 'match':' '.join(match), 'gen_log': dial}
print('Mismatch dialogs:', count)
json.dump(new_data, open(save_path, 'w'), indent=2)