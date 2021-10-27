
# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
from convlab2.nlu.jointBERT.multiwoz import BERTNLU
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.nlg.template.multiwoz import TemplateNLG
from model_LsGPT import *
from convlab2.dialog_agent import PipelineAgent, BiSession
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from convlab2.util.analysis_tool.analyzer import Analyzer
#from user_gpt import GPT2Agent
#from pprint import pprint
import random
import numpy as np
import torch
import json

slot_map={'leave':'leaveAt', 'leaveat':'leaveAt', 'arrive':'arriveBy', 'arriveby':'arriveBy',
          'id':'trainID', 'trainid':'trainID', 'car':'car type'}
def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)

def fix_goal(goal):
    for domain in goal:
        for intent, sv in goal[domain].items():
            if isinstance(sv, list):
                for i, slot in enumerate(sv):
                    goal[domain][intent][i]=slot_map.get(slot, slot)
            elif isinstance(sv, dict):
                for key in sv:
                    if key in slot_map:
                        goal[domain][intent][slot_map[key]]=goal[domain][intent].pop(key)
    return goal

def prepare_goal_list():
    data=json.load(open('data/multi-woz-2.1-processed/data_for_damd_fix.json', 'r', encoding='utf-8'))
    test_list = [l.strip().lower()
                     for l in open('data/MultiWOZ_2.1/testListFile.txt', 'r').readlines()]
    goal_list=[]
    for dial_id, dial in data.items():
        if dial_id in test_list:
            goal=dial['goal']
            goal=fix_goal(goal)
            goal_list.append(goal)
    return goal_list


def test_end2end():

    # BERT nlu trained on sys utterance
    user_nlu = BERTNLU(mode='sys', config_file='multiwoz_sys_context.json',
                       model_file='bert_nlu/bert_multiwoz_sys_context.zip')
    # not use dst
    #user_nlu = None
    user_dst = None
    user_policy = RulePolicy(character='usr')
    user_nlg = TemplateNLG(is_user=True)
    user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name='user')


    #user_agent=GPT2Agent(policy=user_policy, name='user_gpt', device=1)
    #sys_agent=turn_level_sys(path='experiments_21/turn-level-DS/best_score_model', device1=2)
    sys_agent=turn_level_sys(path='RL_exp/rl-10-19-use-scheduler/best_DS', device1=0)
    #sys_agent = LsGPT(return_tuple=True,device=1)
    analyzer = Analyzer(user_agent=user_agent, dataset='multiwoz')

    set_seed(20200202)
    #goal_list=prepare_goal_list()
    #dial_nums=len(goal_list)
    dial_nums=1000
    analyzer.comprehensive_analyze(sys_agent=sys_agent, model_name='turn-level-GPT',\
        total_dialog=dial_nums, show_turns=0,save_path='gen_dials_{}.json'.format(dial_nums))

if __name__ == '__main__':
    test_end2end()