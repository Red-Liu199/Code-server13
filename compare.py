import json
from ontology import *
from session import turn_level_session, fix_seed

def goal_trans(goal):
    new_goal={}
    for domain in goal:
        new_goal[domain]={}
        for intent, sv in goal[domain].items():
            if intent=='reqt':
                new_goal[domain][intent]=[normlize_slot_names.get(slot.lower(), slot.lower()) for slot in sv.keys()]
            elif intent in ['info', 'book']:
                new_goal[domain][intent]={}
                for slot, value in sv.items():
                    slot=normlize_slot_names.get(slot.lower(), slot.lower())
                    new_goal[domain][intent][slot]=value
                
    return new_goal

if __name__=='__main__':
    fix_seed()
    DS_path='experiments_21/turn-level-DS/best_score_model'
    US_path='experiments_21/all_turn-level-US-10-7_sd11_lr0.0001_bs4_ga8/best_score_model'
    DS_device=0
    US_device=1
    session=turn_level_session(DS_path, US_path, DS_device, US_device)

    data=json.load(open('gen_dials_1000.json','r', encoding='utf-8'))
    goal_batches=[]
    goal_batch=[]
    count=0
    success=0
    match=0
    # Evaluate the dialogue quality generated from GPT-2 based DS and Agenda-based US
    for dial in data.values():
        goal=goal_trans(dial['goal'])
        gen_dial=dial['log']
        metrics=session.get_metrics(goal, gen_dial)
        success+=metrics[0]
        match+=metrics[1]
        goal_batch.append(goal)
        count+=1
        if len(goal_batch)>=32 or count==len(data):
            goal_batches.append(goal_batch)
            goal_batch=[]
    print(print(len(goal_batches)))
    print('Success:{:.2f}, Match:{:.2f}'.format(success/count, match/count))
    # Evaluate the dialogue quality generated from GPT-2 based DS and US
    # session.validate(init_goal_batches=goal_batches)
    json.dump(goal_batches, open('gen_goals.json', 'w'), indent=2)
    