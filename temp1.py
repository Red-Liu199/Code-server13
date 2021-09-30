from config import global_config as cfg
from eval import MultiWozEvaluator
from reader import MultiWozReader
from transformers import GPT2Tokenizer
import json
import ontology

cfg.rl_trin=True
cfg.fix_data=True
tokenizer=GPT2Tokenizer.from_pretrained('best_model')
reader = MultiWozReader(tokenizer)
path='data/multi-woz-2.1-processed/data_for_damd_fix.json'
data=json.load(open(path,'r', encoding='utf-8'))

def clean_dataset(data):
    count=0
    for dial_id in data:
        dial=data[dial_id]['log']
        for turn_id, turn in enumerate(dial):
            if 'portugese' in turn['constraint']:
                count+=1
                data[dial_id]['log'][turn_id]['constraint']=turn['constraint'].replace('portugese','portuguese')
            if 'portugese' in turn['user']:
                count+=1
                data[dial_id]['log'][turn_id]['user']=turn['user'].replace('portugese','portuguese')
    print(count)
    json.dump(data,open(path, 'w'), indent=2)

def count_end_act(data):
    act_set={}
    total=0
    for dial_id in data:
        dial=data[dial_id]['log']
        turn=dial[-1]
        sys_act=turn['sys_act']
        total+=1

        for word in sys_act.split():
            if word.startswith('['):
                if word not in act_set:
                    act_set[word]=1
                else:
                    act_set[word]+=1
    print('Total turns: ', total)
    print('Actiongs: ', act_set)
    return act_set

def count_turn_domain(data):
    count=0
    for dial_id in data:
        dial=data[dial_id]['log']
        for turn_id, turn in enumerate(dial):
            turn_domain=turn['turn_domain']
            if len(turn_domain.split())>1:
                count+=1
    print(count)

def detect_bs_error(data):
    total=0
    total_turn=0
    special_data={}
    log_num=3
    for dial_id in data:
        dial=data[dial_id]['log']
        flag=0
        for turn_id, turn in enumerate(dial):
            total_turn+=1
            cons=turn['constraint']
            if 'name' in cons:
                cons_dict=reader.bspan_to_constraint_dict(cons)
                for domain in cons_dict:
                    name_value=cons_dict[domain].get('name', None)
                    if name_value and name_value not in turn['user']:# not in the current turn
                        name_in_user=False
                        for i in range(turn_id):
                            if name_value in dial[i]['user']:# in previous turns
                                name_in_user=True
                                break
                        if not name_in_user:
                            total+=1
                            flag=1
                            if log_num>0:
                                print('user:', turn['user'])
                                print('cons:', cons)
                                log_num-=1
        if flag:
            special_data[dial_id]=data[dial_id]
    print(total, total_turn)
    json.dump(special_data, open('special_data.json', 'w'), indent=2)
#detect_bs_error(data)
#count_end_act(data)
#clean_dataset(data)
count_turn_domain(data)
