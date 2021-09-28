
# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
from convlab2.nlu.jointBERT.multiwoz import BERTNLU
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.nlg.template.multiwoz import TemplateNLG
#from convlab2.e2e.damd.multiwoz import Damd
from model_LsGPT import *
from convlab2.dialog_agent import PipelineAgent, BiSession
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from convlab2.util.analysis_tool.analyzer import Analyzer
from user_gpt import GPT2Agent
from pprint import pprint
import random
import numpy as np
import torch


def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)


def test_end2end():

    # BERT nlu trained on sys utterance
    #user_nlu = BERTNLU(mode='sys', config_file='multiwoz_sys_context.json',
    #                   model_file='bert_nlu/bert_multiwoz_sys_context.zip')
    # not use dst
    #user_nlu = None
    #user_dst = None
    user_policy = RulePolicy(character='usr')
    #user_nlg = TemplateNLG(is_user=True)
    #user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name='user')


    user_agent=GPT2Agent(policy=user_policy, name='user_gpt', device=1)
    sys_agent = LsGPT(return_tuple=True,device=1)

    analyzer = Analyzer(user_agent=user_agent, dataset='multiwoz')

    set_seed(20)
    dial_nums=10
    analyzer.comprehensive_analyze(sys_agent=sys_agent, model_name='LS-GPT',\
        total_dialog=dial_nums, show_turns=0,save_path='gen_dials_{}.json'.format(dial_nums))

if __name__ == '__main__':
    test_end2end()