from config import global_config as cfg
from eval import MultiWozEvaluator
from reader import MultiWozReader
from transformers import GPT2Tokenizer
import json
import ontology

tokenizer=GPT2Tokenizer.from_pretrained('experiments_21/turn-level-DS/best_score_model')
reader = MultiWozReader(tokenizer)
evaluator = MultiWozEvaluator(reader)

def compare_offline_result(path1, path2):
    succ1_unsuc2=[]
    succ2_unsuc1=[]
    data1=json.load(open(path1, 'r', encoding='utf-8'))
    data2=json.load(open(path2, 'r', encoding='utf-8'))
    dials1=evaluator.pack_dial(data1)
    dials2=evaluator.pack_dial(data2)
    counts = {}
    for req in evaluator.requestables:
        counts[req+'_total'] = 0
        counts[req+'_offer'] = 0
    for dial_id in dials1:
        dial1=dials1[dial_id]
        dial2=dials2[dial_id]
        reqs = {}
        goal = {}
        if '.json' not in dial_id and '.json' in list(evaluator.all_data.keys())[0]:
            dial_id = dial_id + '.json'
        for domain in ontology.all_domains:
            if evaluator.all_data[dial_id]['goal'].get(domain):
                true_goal = evaluator.all_data[dial_id]['goal']
                goal = evaluator._parseGoal(goal, true_goal, domain)
        # print(goal)
        for domain in goal.keys():
            reqs[domain] = goal[domain]['requestable']

        # print('\n',dial_id)
        success1, match1, _, _ = evaluator._evaluateGeneratedDialogue(dial1, goal, reqs, counts)
        success2, match2, _, _ = evaluator._evaluateGeneratedDialogue(dial2, goal, reqs, counts)
        if success1 and not success2:
            succ1_unsuc2.append(dial_id)
        elif success2 and not success1:
            succ2_unsuc1.append(dial_id)
    print('Success in data1 and unsuccess in data2:', len(succ1_unsuc2), succ1_unsuc2)
    print('Success in data2 and unsuccess in data1:', len(succ2_unsuc1), succ2_unsuc1)


def compare_online_result(path1, path2):
    succ1_unsuc2=[]
    succ2_unsuc1=[]
    data1=json.load(open(path1, 'r', encoding='utf-8'))
    data2=json.load(open(path2, 'r', encoding='utf-8'))
    counts = {}
    for req in evaluator.requestables:
        counts[req+'_total'] = 0
        counts[req+'_offer'] = 0
    flag1=0
    flag2=0
    for i, dial_id in enumerate(reader.test_list):
        reqs = {}
        goal = {}
        dial1=data1[i]
        dial2=data2[i]
        if isinstance(dial1, list):
            data1[i]={dial_id:dial1}
            flag1=1
        elif isinstance(dial1, dict):
            dial1=dial1[dial_id]
        
        if isinstance(dial2, list):
            data2[i]={dial_id:dial2}
            flag2=1
        elif isinstance(dial2, dict):
            dial2=dial2[dial_id]

        init_goal=reader.data[dial_id]['goal']
        for domain in ontology.all_domains:
            if init_goal.get(domain):
                true_goal = init_goal
                goal = evaluator._parseGoal(goal, true_goal, domain)
        for domain in goal.keys():
            reqs[domain] = goal[domain]['requestable']
        success1, match2, _, _ = evaluator._evaluateGeneratedDialogue(dial1, goal, reqs, counts)
        success2, match2, _, _ = evaluator._evaluateGeneratedDialogue(dial2, goal, reqs, counts)
        if success1 and not success2:
            succ1_unsuc2.append(dial_id)
        elif success2 and not success1:
            succ2_unsuc1.append(dial_id)
    print('Success in data1 and unsuccess in data2:', len(succ1_unsuc2), succ1_unsuc2)
    print('Success in data2 and unsuccess in data1:', len(succ2_unsuc1), succ2_unsuc1)
    if flag1:
        json.dump(data1, open(path1, 'w'), indent=2)
    if flag2:
        json.dump(data2, open(path2, 'w'), indent=2)

def group_act(act):
    for domain in act:
        for intent, sv in act[domain].items():
            act[domain][intent]=set(sv)
    return act

def find_unseen_act(path1=None, path2=None):
    data=json.load(open('data/multi-woz-2.1-processed/data_for_us.json', 'r', encoding='utf-8'))
    train_act_pool=[]
    unseen_act_pool=[]
    unseen_dials=[]
    for dial_id, dial in data.items():
        if dial_id in reader.train_list:
            for turn in dial:
                user_act=reader.aspan_to_act_dict(turn['usr_act'], 'user')
                user_act=group_act(user_act)
                if user_act not in train_act_pool:
                    train_act_pool.append(user_act)
    for dial_id, dial in data.items():
        if dial_id in reader.test_list:# or dial_id in reader.dev_list:
            unseen_turns=0
            for turn in dial:
                user_act=reader.aspan_to_act_dict(turn['usr_act'], 'user')
                user_act=group_act(user_act)
                if user_act not in train_act_pool:
                    unseen_act_pool.append(user_act)
                    unseen_turns+=1
            if unseen_turns>0:
                unseen_dials.append(dial_id)
    print('Total training acts:', len(train_act_pool), 'Unseen acts:', len(unseen_act_pool))
    print('Unseen dials:',len(unseen_dials))
    if path1 and path2:
        data1=json.load(open(path1, 'r', encoding='utf-8'))
        data2=json.load(open(path2, 'r', encoding='utf-8'))
        unseen_act_pool1=[]
        unseen_act_pool2=[]
        for dial1, dial2 in zip(data1, data2):
            dial1=list(dial1.values())[0]
            dial2=list(dial2.values())[0]
            for turn in dial1:
                user_act=reader.aspan_to_act_dict(turn['usr_act'], 'user')
                user_act=group_act(user_act)
                if user_act not in train_act_pool:
                    unseen_act_pool1.append(user_act)
            for turn in dial2:
                user_act=reader.aspan_to_act_dict(turn['usr_act'], 'user')
                user_act=group_act(user_act)
                if user_act not in train_act_pool:
                    unseen_act_pool2.append(user_act)
        print('Unseen acts in path1:', len(unseen_act_pool1))
        print('Unseen acts in path2:', len(unseen_act_pool2))
    return unseen_dials

def extract_goal():
    data=json.load(open('data/multi-woz-2.1-processed/data_for_damd_fix.json', 'r', encoding='utf-8'))
    goal_list=[]
    for dial_id, dial in data.items():
        goal=dial['goal']
        goal_list.append(goal)
    json.dump(goal_list, open('analysis/goals.json', 'w'), indent=2)



if __name__=='__main__':
    path1='experiments_21/turn-level-DS/best_score_model/result.json'
    path2='RL_exp/rl-10-19-use-scheduler/best_DS/result.json'
    #compare_offline_result(path1, path2)
    path1='experiments_21/turn-level-DS/best_score_model/validate_result.json'
    path2='RL_exp/rl-10-19-use-scheduler/best_DS/validate_result.json'
    #compare_online_result(path1, path2)
    #bspn='[restaurant] pricerange expensive area west'
    #print(reader.bspan_to_DBpointer(bspn, ['restaurant']))
    #unseen_dials=find_unseen_act(path1, path2)
    #print(unseen_dials)
    #act='[taxi] [inform] destination cambridge train station [taxi] [request] car'
    #print(reader.aspan_to_act_dict(act, 'user'))
    #print(set(reader.aspan_to_act_dict(act, 'user')))
    extract_goal()
