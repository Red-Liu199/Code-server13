import json
slot_list_act=[]
intent_list=[]
from reader import MultiWozReader
from transformers import GPT2Tokenizer
import json
import ontology
tokenizer=GPT2Tokenizer.from_pretrained('experiments_21/turn-level-DS/best_score_model')
reader = MultiWozReader(tokenizer)

def act_dict_to_aspn(act):
    #<sos_a> [restaurant] [offerbooked]  reference [general] [reqmore] <eos_a>
    slot_map={'post':'postcode','addr':'address','ref':'reference','dest':'destination','depart':'departure', 'fee':'price', 'entrance fee':'price'}
    act_list=[]
    for key in act:
        domain=key.split('-')[0].lower()
        intent=key.split('-')[1].lower()
        act_list.append('['+domain+']')
        act_list.append('['+intent+']')
        if intent not in intent_list:
            intent_list.append(intent)
        if domain !='general':
            if intent=='inform':
                for item in act[key]:
                    slot=item[0].lower()
                    slot=slot_map.get(slot,slot)
                    act_list.append(slot)#slot
                    act_list.append(item[1].lower())#value
                    if slot not in slot_list_act:
                        #slot_list_act.append(item[0])
                        slot_list_act.append(slot)
            elif intent=='request':
                for item in act[key]:
                    slot=item[0].lower()
                    slot=slot_map.get(slot,slot)
                    act_list.append(slot)#slot
                    if slot not in slot_list_act:
                        slot_list_act.append(slot)
    aspn=' '.join(act_list)         
    return aspn

def goal_to_gpan(goal, slot_list_goal=None):
    slot_list_goal=[] if slot_list_goal==None else slot_list_goal
    token_map={'info':'[inform]','reqt':'[request]','fail_info':'[fail_info]','book':'[book]','fail_book':'[fail_book]'}
    goal_list=[]
    for domain in goal:
        goal_list.append('['+domain+']')
        for intent in goal[domain]:
            if intent in ['fail_book','fail_info']:
                # we do not consider the failing condition 
                continue
            goal_list.append(token_map.get(intent,''))
            if isinstance(goal[domain][intent],dict):
                for slot,value in goal[domain][intent].items():
                    slot=slot.lower()
                    value=value.lower() if isinstance(value, str) else value
                    if slot in ['pre_invalid','invalid']:
                        continue
                        '''
                        if slot=='pre_invalid':
                            goal_list.append('[pre_invalid]')
                        elif slot=='invalid':
                            goal_list.append('[invalid]')
                            value='yes' if value==True else 'no'
                            goal_list.append(value)
                        '''
                    else:
                        # some special case
                        if slot=='trainid':
                            slot='id'
                        if slot=='car type':
                            slot='car'
                        if slot in ['entrance fee', 'fee']:
                            slot='price'
                        if slot=='duration':
                            slot='time'
                        slot='arrive' if slot=='arriveby' else slot
                        slot='leave' if slot=='leaveat' else slot
                        goal_list.append(slot)
                        # some request goal may be in the form of dict
                        if value!='' and value!='?':
                            goal_list.append(str(value))
                    if slot not in slot_list_goal:
                        slot_list_goal.append(slot)
            elif isinstance(goal[domain][intent],list):
                for slot in goal[domain][intent]:
                    slot=slot.lower()
                    goal_list.append(slot)
                    if slot not in slot_list_goal:
                        slot_list_goal.append(slot)
    gpan=' '.join(goal_list)
    #print(slot_list_goal)
    return gpan

def count(data):
    intent_pool=[]
    slot_pool=[]
    goal_pool={}
    no_act_count=0
    for dial_id in data:
        goal=data[dial_id]['goal']
        if dial_id=='mul2168.json':
            t=1
        for domain in goal:
            if domain in ['message','topic']:
                continue
            if domain not in goal_pool:
                goal_pool[domain]={}
            for intent in goal[domain]:
                if intent not in goal_pool[domain]:
                    goal_pool[domain][intent]=[]
                if isinstance(goal[domain][intent],dict):
                    for slot in goal[domain][intent].keys():
                        if slot not in goal_pool[domain][intent]:
                            goal_pool[domain][intent].append(slot)
                elif isinstance(goal[domain][intent],list):
                    for slot in goal[domain][intent]:
                        if slot not in goal_pool[domain][intent]:
                            goal_pool[domain][intent].append(slot)
        '''
        for turn_id,turn in enumerate(data[dial_id]['log']):
            if turn_id%2==0:#user
                if 'dialog_act' not in turn:
                    no_act_count+=1
                    continue
                usr_act=turn['dialog_act']
                for key in usr_act:
                    if key not in intent_pool:
                        intent_pool.append(key)
                    for item in usr_act[key]:
                        if item[0] not in slot_pool:
                            slot_pool.append(item[0])
        '''

    print(goal_pool)
    #print('user turns without dialog act:', no_act_count)




if __name__ == "__main__":
    path1='data/multi-woz-2.1-processed/data_for_damd_fix.json'
    path2='data/MultiWOZ_2.1/data.json'
    save_path='data/multi-woz-2.1-processed/data_for_us.json'
    data1=json.load(open(path1,'r', encoding='utf-8'))
    data2=json.load(open(path2,'r', encoding='utf-8'))
    #count(data1)
    slot_list_goal=[]
    new_data={}
    for dial_id in data1:
        dial_id_up=dial_id.split('.')[0].upper()+'.json'
        dial1=data1[dial_id]
        dial2=data2[dial_id_up]
        new_data[dial_id]={}
        goal=dial1['goal']
        goal=reader.goal_norm(goal)
        new_data[dial_id]=[]
        pv_user_act=None
        pv_constraint=None
        for turn_id, turn in enumerate(dial1['log']):
            if pv_user_act is not None:
                goal=reader.update_goal(goal, pv_user_act, pv_constraint)
            turn_domain=turn['turn_domain'].split()
            cur_domain=turn_domain[0] if len(turn_domain)==1 else turn_domain[1]
            cur_domain=cur_domain[1:-1] if cur_domain.startswith('[') else cur_domain
            gpan=reader.goal_to_gpan(goal, cur_domain)
            entry={}
            entry['goal']=gpan
            for field in ['user','resp','constraint','sys_act','turn_domain', 'turn_num']:
                entry[field]=turn[field]
            if 'dialog_act' in dial2['log'][2*turn_id]:
                entry['usr_act']=act_dict_to_aspn(dial2['log'][2*turn_id]['dialog_act'])
            else:
                entry['usr_act']=''
            pv_user_act=reader.aspan_to_act_dict(entry['usr_act'], side='user')
            pv_constraint=reader.bspan_to_constraint_dict(entry['constraint'])
            new_data[dial_id].append(entry)

    json.dump(new_data, open(save_path, 'w'), indent=2)
    
    
