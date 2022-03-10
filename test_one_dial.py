from config import global_config as cfg
from eval import MultiWozEvaluator
from reader import MultiWozReader
from transformers import GPT2Tokenizer
import json
import ontology
from session import turn_level_session

cfg.rl_trin=True
tokenizer=GPT2Tokenizer.from_pretrained('best_model')
reader = MultiWozReader(tokenizer)
evaluator = MultiWozEvaluator(reader)
'''
dial_id='pmul3162.json'
path='experiments_21/DS_base/best_score_model/offline_mismatch.json'
data=json.load(open(path,'r', encoding='utf-8'))

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
print('Goal:\n', goal)
for domain in goal.keys():
    reqs[domain] = goal[domain]['requestable']
print('Reqt:\n', reqs)
#print('Dial:\n', dial)
success, match, _, _ = evaluator._evaluateGeneratedDialogue(dial, goal, reqs, counts, soft_acc=True)
print('Success:{}, match:{}'.format(success, match))
#bspn="<sos_b> [restaurant]  food british area centre pricerange dontcare <eos_b>"
bspn="<sos_b> [restaurant]  food british area centre <eos_b>"
constraint_dict = reader.bspan_to_constraint_dict(bspn)
print('constraint_dict\n',constraint_dict)
domain='restaurant'
venues = reader.db.queryJsons(domain, constraint_dict[domain], return_name=True)
print('venues:', venues)
'''
ua='<sos_a> [restaurant] [nooffer]  food [recommend]  name [inform]  food choice price area <eos_a>'
#print(reader.aspan_to_act_list(ua))
print(reader.aspan_to_act_dict(ua, side='sys'))
'''
path1='/home/liuhong/myworkspace/experiments_21/turn-level-DS/best_score_model'
path2='/home/liuhong/myworkspace/experiments_21/all_turn-level-us-111_sd11_lr0.0001_bs8_ga4/best_score_model'
session=turn_level_session(path1, path2, 2,2)
act_list=[
        "<sos_ua> [restaurant] [inform]  time 18:00 day monday people 7 <eos_ua>",
        "<sos_ua> [restaurant] [inform]  time 18:00 day monday people 7 [restaurant] [request]  reference <eos_ua>",
        "<sos_ua> [restaurant] [inform]  time 19:00 day monday people 7 <eos_ua>",
        "<sos_ua> [restaurant] [inform]  price expensive time 18:00 day monday people 7 <eos_ua>",
        "<sos_ua> [restaurant] [inform]  price do nt care time 18:00 day monday people 7 <eos_ua>",
        "<sos_ua> [restaurant] [inform]  price do nt care time 18:00 day monday people 7 [restaurant] [request]  reference <eos_ua>",
        "<sos_ua> [restaurant] [inform]  time 19:00 day monday people 7 [restaurant] [request]  reference <eos_ua>",
        "<sos_ua> [restaurant] [inform]  price expensive time 18:00 day monday people 7 [restaurant] [request]  reference <eos_ua>",
        "<sos_ua> [restaurant] [inform]  time 18:00 day tuesday people 7 <eos_ua>",
        "<sos_ua> [restaurant] [inform]  time 18:00 day saturday people 7 <eos_ua>"
      ]
pv_aspn='<sos_a> [restaurant] [inform]  food choice area [request]  price <eos_a>'
gpan='"<sos_g> [taxi] [inform] arrive 18:00 [request] car phone [attraction] [inform] name holy trinity church [request] price [restaurant] [book] people 7 day monday time 18:00 <eos_g>",'
goal={'taxi':{'inform':{'arrive':'18:00'}, 'request':['car', 'phone']},
      'attraction':{'inform':{'name':'holy trinity church'}, 'request':['price']},
      'restaurant':{'book':{'people':'7', 'day':'monday', 'time':'18:00'}}}
print(session.find_best_usr_act(act_list, goal, pv_aspn))
'''