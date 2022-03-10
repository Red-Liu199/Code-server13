from transformers import GPT2Tokenizer, GPT2Model
from reader import MultiWozReader
import torch
import os, json
tokenizer=GPT2Tokenizer.from_pretrained('best_model')
reader = MultiWozReader(tokenizer)
def integrate_book_slot(data):
    book_pool={}
    inform_pool={}
    for dial_id, dial in data.items():
        goal=dial['goal']
        for domain in goal:
            if 'book' in goal[domain]:
                for slot in goal[domain]['book']:
                    if slot in ['invalid', 'pre_invalid']:
                        continue
                    else:
                        if domain not in book_pool:
                            book_pool.update({domain:[slot]})
                        elif slot not in book_pool[domain]:
                            book_pool[domain].append(slot)
            if 'info' in goal[domain]:
                for slot in goal[domain]['info']:
                    if slot in ['invalid', 'pre_invalid']:
                        continue
                    else:
                        if domain not in inform_pool:
                            inform_pool.update({domain:[slot]})
                        elif slot not in inform_pool[domain]:
                            inform_pool[domain].append(slot)
    print(book_pool)
    print(inform_pool)
if __name__=='__main__':
    '''
    tokenizer = GPT2Tokenizer.from_pretrained('../distilgpt2')
    model = GPT2Model.from_pretrained('../distilgpt2')

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    outputs = model(**inputs, output_attentions=True, return_dict=True)
    print(outputs['attentions'][0][0][0])
    
    data=json.load(open('data/multi-woz-2.1-processed/data_for_us.json', 'r'))
    goal_pool=[]
    act_pool={
        'empty':[],
        'others':[]
    }
    count1, count2 = 0, 0
    for dial_id, dial in data.items():
        goal, user_act = dial[-1]['goal'], dial[-1]['usr_act']
        if goal=="":
            count1+=1
            if user_act not in act_pool['empty']:
                act_pool['empty'].append(user_act)
        else:
            count2+=1
            if user_act not in act_pool['others']:
                act_pool['others'].append(user_act)
    print('Total dials:', len(data))
    print('Empty goal in last turn:', count1, 'others:', count2)
    print('User act of empty goal:', len(act_pool['empty']), act_pool['empty'])
    #print('User act of no empty goal:', len(act_pool['others']), act_pool['others'])
    
    #data=json.load(open('data/multi-woz-2.1-processed/data_for_damd_fix.json', 'r'))
    #integrate_book_slot(data)
    
    data=json.load(open('data/multi-woz-2.1-processed/data_for_us.json', 'r'))
    pool=[]
    for dial_id, dial in data.items():
        goal=dial[0]['goal']
        for word in goal.split():
            if word.startswith('[') and word not in pool:
                pool.append(word)
    print(pool)
    '''
    data=json.load(open('data/multi-woz-2.1-processed/data_for_us.json', 'r'))
    count1, count2 = 0, 0
    for dial_id, dial in data.items():
        for turn_num, turn in enumerate(dial):
            goal=reader.aspan_to_act_dict(turn['goal'], side='user')
            user_act=reader.aspan_to_act_dict(turn['usr_act'], side='user')
            sys_act=reader.aspan_to_act_dict(turn['sys_act'], side='sys')
            for domain in sys_act:
                if domain not in goal:
                    continue
                if 'inform' not in sys_act[domain]:
                    continue
                if 'request' not in goal[domain]:
                    continue
                for slot in sys_act[domain]['inform']:
                    if slot not in goal[domain]['request']:
                        continue
                    if domain not in user_act or 'request' not in user_act[domain] or slot not in user_act[domain]['request']:
                        #print((dial_id, turn_num, slot))
                        count1+=1
                        if turn_num<len(dial)-1:
                            next_goal=reader.aspan_to_act_dict(dial[turn_num+1]['goal'], side='user')
                            if domain not in next_goal or 'request' not in next_goal[domain] or slot not in next_goal[domain]['request']:
                                print((dial_id, turn_num, slot))
                                count2+=1
    print(count1, count2)

            



        

            
