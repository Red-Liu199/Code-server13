'''
import json
path='temp.json'
data=json.load(open(path,'r', encoding='utf-8'))
slot1={}
slot2={}
for domain in data['MultiWOZ_dict']:
    if domain not in slot1:
        slot1[domain]=[]
    for intent in data['MultiWOZ_dict'][domain]:
        for slot in data['MultiWOZ_dict'][domain][intent]:
            if slot not in slot1[domain]:
                slot1[domain].append(slot)

for domain in data['ConvLab2_dict']:
    if domain not in slot2:
        slot2[domain]=[]
    for intent in data['ConvLab2_dict'][domain]:
        for slot in data['ConvLab2_dict'][domain][intent]:
            if slot not in slot2[domain]:
                slot2[domain].append(slot)
print(slot1)
print(slot2)
'''
import json
from transformers import GPT2Tokenizer
from config import global_config as cfg
from reader import MultiWozReader
tokenizer=GPT2Tokenizer.from_pretrained('best_model')
reader = MultiWozReader(tokenizer)
path='data/multi-woz-2.1-processed/data_for_damd.json'
data=json.load(open(path,'r', encoding='utf-8'))
intent_list=[]
slot_dict={}
intent_dict1={}
for dial in data:
    for turn in data[dial]['log']:
        act=turn['sys_act']
        action_list=reader.aspan_to_act_list(act)#[domain-intent-slot]
        for action in action_list:
            action=action.split('-')
            if action[0] not in intent_dict1:
                intent_dict1[action[0]]={}
            if action[1] not in intent_dict1[action[0]]:
                intent_dict1[action[0]][action[1]]=[]
            if action[2] not in intent_dict1[action[0]][action[1]]:
                intent_dict1[action[0]][action[1]].append(action[2])




intent_path='../ConvLab-2/convlab2/nlu/jointBERT/multiwoz/data/sys_data/intent_vocab.json'
tag_path='../ConvLab-2/convlab2/nlu/jointBERT/multiwoz/data/sys_data/tag_vocab.json'
intent_vacab=json.load(open(intent_path,'r', encoding='utf-8'))
tag_vacab=json.load(open(tag_path,'r', encoding='utf-8'))
#print('total_intent:', len(intent_vacab))
intent_dict={}
act_list=[]
for intent in intent_vacab:
    domain=intent.split('-')[0]
    act_sv=intent.split('-')[1]
    act=act_sv.split('+')[0]
    if act not in act_list:
        act_list.append(act)
    sv=act_sv.split('+')[1]
    slot=sv.split('*')[0]
    if domain not in intent_dict:
        intent_dict[domain]={}
    if act not in intent_dict[domain]:
        intent_dict[domain][act]=[]
    if slot not in intent_dict[domain][act]:
        intent_dict[domain][act].append(slot)

for intent in tag_vacab[1:]:
    domain=intent.split('-')[1]
    act_s=intent.split('-')[2]
    act=act_s.split('+')[0]
    slot=act_s.split('+')[1]
    if domain not in intent_dict:
        intent_dict[domain]={}
    if act not in intent_dict[domain]:
        intent_dict[domain][act]=[]
    if slot not in intent_dict[domain][act]:
        intent_dict[domain][act].append(slot)
dict={}
dict['MultiWOZ_dict']=intent_dict1
dict['ConvLab2_dict']=intent_dict
json.dump(dict, open('onto.json', 'w'), indent=2)

