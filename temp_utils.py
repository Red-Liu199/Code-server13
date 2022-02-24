from config import global_config as cfg
from eval import MultiWozEvaluator
from reader import MultiWozReader
from transformers import GPT2Tokenizer
import json
import ontology
import numpy as np
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

cfg.rl_trin=True
cfg.fix_data=True
tokenizer=GPT2Tokenizer.from_pretrained('best_model')
reader = MultiWozReader(tokenizer)
path='data/multi-woz-2.1-processed/data_for_damd_fix.json'
data=json.load(open(path,'r', encoding='utf-8'))

vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
              "potato", "wheat", "barley"]
farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
           "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])

def heat_map():
    fig, ax = plt.subplots()
    im = ax.imshow(harvest)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(farmers)), labels=farmers)
    ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(vegetables)):
        for j in range(len(farmers)):
            text = ax.text(j, i, harvest[i, j],
                        ha="center", va="center", color="w")

    ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    plt.savefig('analysis/temp.png')

def clean_dataset(data):
    count1, count2, count3=0, 0, 0
    for dial_id in data:
        dial=data[dial_id]['log']
        for turn_id, turn in enumerate(dial):
            flag=0
            if 'portugese' in turn['constraint']:
                flag=1
                count1+=1
                data[dial_id]['log'][turn_id]['constraint']=turn['constraint'].replace('portugese','portuguese')
            if 'portugese' in turn['user']:
                flag=1
                count2+=1
                data[dial_id]['log'][turn_id]['user']=turn['user'].replace('portugese','portuguese')
            if flag:
                count3+=1

    print('Errors in user:',count2)
    print('Errors in constraint:',count1)
    print('Errors turn:', count3)
    #json.dump(data,open(path, 'w'), indent=2)

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
    count=0
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
        if flag:
            count+=1
            special_data[dial_id]=data[dial_id]
    print(total, total_turn, count)
    json.dump(special_data, open('special_data.json', 'w'), indent=2)

def extract_score(path):
    extract_type=['match rate', 'success rate']
    extract_result={}
    for key in extract_type:
        extract_result[key]=[]
    with open(path, 'r') as f:
        for line in f.readlines():
            if line.startswith('INFO:root:Avg_US_reward:'):
                for key in extract_result:
                    temp=line[line.index(key):]
                    value=temp[temp.index(':')+1:temp.index(',')]
                    extract_result[key].append(float(value))
    return extract_result


if __name__=='__main__':
    #detect_bs_error(data)
    #count_end_act(data)
    #clean_dataset(data)

    #count_turn_domain(data)
    #path1='log21/log_train_all_RL-old-ds-new-us-only-us_sd11.txt'
    path1='/home/liuhong/myworkspace/log21/log_train_RL-1-10-beam-1_sd11.txt'
    #path2='log21/log_train_RL-12-30_sd11.txt'
    score1=extract_score(path1)
    #score2=extract_score(path2)
    data=score1['success rate']
    plt.plot(data)
    plt.savefig('analysis/success.png')
    #print(data)
    #path='/mnt/workspace/liuhong/VLS-GPT/MultiWOZ-exp/data/multi-woz-2.1-processed/data_for_damd.json'
    #data0=json.load(open(path,'r', encoding='utf-8'))
    #clean_dataset(data0)
    #reader.fix_dialog_state(data0)
    #heat_map()
