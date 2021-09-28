import os
import zipfile
import torch

from convlab2.util.file_util import cached_path
import transformers
from config import global_config as cfg
from reader import MultiWozReader
import random
from collections import Counter
import re
import json
import copy
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from convlab2.dialog_agent import Agent
from convlab2.policy import Policy
from convlab2.policy.rule.multiwoz import RulePolicy
from prepare_us_data import goal_to_gpan
from utils import modify_map_file
default_model_path='experiments_21/US_base/best_loss_model'
slot_list_goal=[]
slot_list=['type', 'price', 'parking', 'stay', 'day', 'people', 'none', 'postcode', 'address', 
           'area', 'stars', 'internet', 'reference', 'destination', 'departure', 'arrive', 'leave', 
           'time', 'ticket', 'phone', 'food', 'name', 'department', 'fee', 'id', 'car']# slots in user act
domain_list=['[restaurant]', '[hotel]', '[attraction]', '[train]', '[taxi]', '[police]', '[hospital]', '[general]']
intent_list=['[inform]','[request]','[bye]', '[thank]', '[greet]']
special_slot_map={'address':'Addr', 'postcode':'Post','reference':'Ref','departure':'Depart','destination':'Dest'}
class GPT2Agent(Agent):

    def __init__(self, policy: Policy, name: str, reader, model_path=default_model_path, device=0):
        # we use policy only for generate goal
        super(GPT2Agent, self).__init__(name=name)
        self.reader=reader
        self.goal_pointer=0
        self.policy=policy
        self.action_history = []
        self.aspn_history=[]
        modify_map_file(model_path)
        self.tokenizer=GPT2Tokenizer.from_pretrained(model_path)
        self.get_special_id()
        self.model=GPT2LMHeadModel.from_pretrained(model_path)
        self.nlu=self.model # model contains nlu
        if cfg.cuda and torch.cuda.is_available(): self.model = self.model.to(device) 
        self.train_list=copy.deepcopy(self.reader.train_list)

        #self.init_session()

    def get_special_id(self):
        self.sos_ua_id=self.tokenizer.convert_tokens_to_ids('<sos_ua>')
        self.eos_ua_id=self.tokenizer.convert_tokens_to_ids('<eos_ua>')
        self.sos_u_id=self.tokenizer.convert_tokens_to_ids('<sos_u>')
        self.eos_u_id=self.tokenizer.convert_tokens_to_ids('<eos_u>')
        self.sos_r_id=self.tokenizer.convert_tokens_to_ids('<sos_r>')
        self.eos_r_id=self.tokenizer.convert_tokens_to_ids('<eos_r>')
        self.sos_g_id=self.tokenizer.convert_tokens_to_ids('<sos_g>')
        self.eos_g_id=self.tokenizer.convert_tokens_to_ids('<eos_g>')
        

    def response(self, observation):
        if isinstance(observation, tuple):
            observation, self.input_action=observation[0], observation[1]
        ua_max_len=20
        user_max_len=60
        if observation=='':
            model_input=self.history_input
        else:
            #transformers 3.5: space needed
            model_input=self.history_input + [self.sos_r_id] + self.tokenizer.encode(' '+observation+' ') + [self.eos_r_id]
        with torch.no_grad():
            #generate user act
            input_ids=model_input + [self.sos_ua_id]
            max_len=1024-ua_max_len
            if len(input_ids)>max_len:
                input_ids=input_ids[-max_len:]
            context_length=len(input_ids)
            outputs = self.model.generate(input_ids=torch.tensor([input_ids]).to(self.model.device),
                                        pad_token_id=self.tokenizer.eos_token_id,
                                        max_length=context_length + ua_max_len, eos_token_id=self.eos_ua_id)
            generated = outputs[0].cpu().numpy().tolist()
            self.ua = self.tokenizer.decode(generated[context_length-1:])#<sos_ua>...<eos_ua>
            self.aspn_history.append(self.ua)
            self.action_history.append(self.get_out_da())
            #generate user utterance
            input_ids=generated + [self.sos_u_id]
            max_len=1024-user_max_len
            if len(input_ids)>max_len:
                input_ids=input_ids[-max_len:]
            context_length=len(input_ids)
            outputs = self.model.generate(input_ids=torch.tensor([input_ids]).to(self.model.device),
                                        pad_token_id=self.tokenizer.eos_token_id,
                                        max_length=context_length + user_max_len, eos_token_id=self.eos_u_id)
            generated = outputs[0].cpu().numpy().tolist()
            user = self.tokenizer.decode(generated[context_length-1:])#<sos_u>...<eos_u>
            response=self.tokenizer.decode(generated[context_length:-1])
            self.history_input=generated#<sos_g>...<eos_g>...<sos_u>...<eos_u>
        
        
        return response

    def is_terminated(self):
        if '[bye]' in self.ua or '[welcome]' in self.ua or '[thank]' in self.ua:
            return True
        else:
            return False

    def get_reward(self):
        return None

    def init_session(self, **kwargs):
        self.policy.init_session()
        if cfg.goal_from_data:
            if self.goal_pointer>=len(self.train_list):
                self.goal_pointer=self.goal_pointer%len(self.train_list)
                print('Goal list index out of range!')
            dial_id=self.train_list[self.goal_pointer]# select a random goal
            self.goal_dict=self.reader.data[dial_id]['goal']
            self.goal_pointer+=1
        else:
            self.goal_dict=self.policy.get_goal()
        if 'goal' in kwargs:
            self.goal_dict=kwargs['goal']
        self.goal=goal_to_gpan(self.goal_dict)
        self.history_input = [self.sos_g_id] + self.tokenizer.encode(' ' + self.goal)+[self.eos_g_id]
        self.action_history=[]
        self.aspn_history=[]
        self.input_action=[]
        #print('user goal:\n', goal_to_gpan(self.goal))
    
    def get_out_da(self):# get the output user act 
        action=[]
        domain=''
        intent=''
        slot=''
        value=[]
        for word in self.ua.split():
            if word=='none':
                t=1
            if word in ['<sos_ua>', '<eos_ua>']:
                continue

            if word in domain_list+intent_list+slot_list and value!=[]:
                entry=['Inform', domain.capitalize(), slot.capitalize(), ' '.join(value)]
                action.append(entry)
                value=[]

            if word in domain_list:
                domain=word[1:-1]
            elif word in intent_list:
                intent=word[1:-1]
                if word in ['[bye]', '[thank]', '[greet]']:
                    entry=[intent, 'general', 'none', 'none']
                    action.append(entry)
            elif word in slot_list:
                if slot=='none' and word=='none':# none is in the slot list
                    entry=[intent.capitalize(), domain.capitalize(), 'none', 'none']
                    action.append(entry)
                slot=special_slot_map.get(word, word)
                if domain=='attraction' and slot=='price':
                    slot='fee'
                if domain=='train' and slot=='price':
                    slot='ticket'
                if intent=='request':
                    entry=['Request', domain.capitalize(), slot.capitalize(), '?']
                    action.append(entry)
            else:#value or none
                value.append(word)
        if value!=[]:
            entry=['Inform', domain.capitalize(), slot.capitalize(), ' '.join(value)]
            action.append(entry)
        #if action==[]:
        #    print(self.ua)
        return action

    def get_in_da(self):# get the input sys act
        return self.input_action
        

        


if __name__ == '__main__':
    user_policy = RulePolicy(character='usr')
    user_agent = GPT2Agent(user_policy, name='user')
    for j in [random.randint(1,100000) for _ in range(20)]:
        random.seed(j)
        user_agent.init_session()
    #print(slot_list_goal)
