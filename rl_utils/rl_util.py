def fix_act(act_dict):
    reqt_set=set(['phone', 'address', 'postcode'])
    reward=0
    for domain in act_dict:
        for intent in act_dict[domain]:
            if intent=='inform':
                inform_set=set(act_dict[domain]['inform'])
                if len(reqt_set&inform_set)>0:
                    inform_set=inform_set|reqt_set
                    reward=1
                act_dict[domain]['inform']=list(inform_set)
    return act_dict,reward