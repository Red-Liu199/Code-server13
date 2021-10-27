# -*- coding: utf-8 -*-
# Copyright 2021 Tsinghua SPMI Lab, Author: Hong Liu
# This script turns LS-GPT to a system agent
# It provides two different format of response: lex response or action list
import os
import zipfile
import torch

from convlab2.util.file_util import cached_path
from config import global_config as cfg
from reader import MultiWozReader
import random
from collections import Counter
import re
import json
import copy
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from convlab2.dialog_agent import Agent
from utils import modify_map_file
from rl_utils.rl_util import fix_act
from session import turn_level_session as Session

DEFAULT_MODEL_URL = "experiments_21/DS_base/best_score_model"
DOMAINS = ["[restaurant]", "[hotel]", "[attraction]", "[train]", "[taxi]", "[police]", "[hospital]", "[general]"]
stopwords = ['and','are','as','at','be','been','but','by', 'for','however','if', 'not','of','on','or','so','the','there','was','were','whatever','whether','would']
all_domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'police', 'hospital']
intent_map={'request':'Request', 'inform':'Inform', 'offerbook':'OfferBook', 'nobook':'NoBook',
 'offerbooked':'OfferBooked', 'reqmore':'reqmore', 'bye':'bye', 'welcome':'welcome', 
 'recommend':'Recommend', 'greet':'greet', 'nooffer':'NoOffer', 'select':'Select'}
domain_map={'restaurant':'Restaurant', 'hotel':'Hotel', 'attraction':'Attraction',
 'train':'Train', 'taxi':'Taxi', 'police':'Police', 'hospital':'Hospital'}

act_list=['Request', 'Inform', 'reqmore', 'bye', 'welcome', 'greet', 'OfferBook', 'NoBook', 'Select', 'NoOffer', 'Recommend', 'Book', 'OfferBooked']
special_slot_map={'address':'Addr', 'postcode':'Post','reference':'Ref','departure':'Depart','destination':'Dest'}
path='onto.json'
onto=json.load(open(path,'r', encoding='utf-8'))
convlab2_dict=onto['ConvLab2_dict']
class LsGPT(Agent):

    def __init__(self, model_path=DEFAULT_MODEL_URL, name='LS-GPT', return_act=False, return_tuple=False, device=0):
        super().__init__(name=name)
        self.return_act=return_act
        self.return_tuple=return_tuple
        if not os.path.exists(model_path):
            print('Model does not exist')
        else:
            print('Load model from {}'.format(model_path))
        modify_map_file(model_path)
        self.tokenizer=GPT2Tokenizer.from_pretrained(model_path)
        self.reader = MultiWozReader(self.tokenizer)
        self.sos_u_id=self.tokenizer.convert_tokens_to_ids('<sos_u>')
        self.sos_b_id=self.tokenizer.convert_tokens_to_ids('<sos_b>')
        self.sos_a_id=self.tokenizer.convert_tokens_to_ids('<sos_a>')
        self.sos_r_id=self.tokenizer.convert_tokens_to_ids('<sos_r>')
        self.eos_u_id=self.tokenizer.convert_tokens_to_ids('<eos_u>')
        self.eos_b_id=self.tokenizer.convert_tokens_to_ids('<eos_b>')
        self.eos_a_id=self.tokenizer.convert_tokens_to_ids('<eos_a>')
        self.eos_r_id=self.tokenizer.convert_tokens_to_ids('<eos_r>')
        self.sos_db_id=self.tokenizer.convert_tokens_to_ids('<sos_db>')
        self.eos_db_id=self.tokenizer.convert_tokens_to_ids('<eos_db>')
        self.model =GPT2LMHeadModel.from_pretrained(model_path)
        if cfg.cuda and torch.cuda.is_available(): self.model = self.model.to(device)  #cfg.cuda_device[0]

        self.model.eval()
        

        self.init_session()

    def init_session(self):
        """Reset the class variables to prepare for a new session."""
        self.info_history=[]
        self.constraint_dict = {}
        self.turn_domain = ['[general]']
        self.py_prev = {'pv_resp': None, 'pv_bspn': None, 'pv_aspn':None, 'pv_bsdx':None}
        self.book_state = {'train': False, 'restaurant': False, 'hotel':False}
        self.user=''

    def response(self, usr):
        """
        Generate agent response given user input.

        Args:
            observation (str):
                The input to the agent.
        Returns:
            response (str):
                The response generated by the agent.
        """
        self.user=usr
        prev_text=[]
        for turn in self.info_history:
            for cell in ['user','bspn','db','aspn','resp']:
                #these cells contain special tokens such as <sos_u> and <eos_u>
                prev_text += self.tokenizer.encode(turn[cell])
        context = prev_text + [self.sos_u_id] + self.tokenizer.encode(usr) + [self.eos_u_id]
        # print('usr:', usr)
        new_turn = self.generate(context)
        delex_resp = new_turn['resp'] # delex resp
        lex_resp, value_map=self.lex_resp(new_turn)
        action=self.aspn_to_action(new_turn['aspn'],value_map)
        #print(value_map)
        new_turn['lex_resp']=lex_resp
        new_turn['sys_intent']=action
        resp=delex_resp if cfg.delex_resp else lex_resp
        self.info_history.append(new_turn)
        if self.return_tuple:
            return (resp, action)
        elif self.return_act:
            return action
        else:
            return resp

    
    def lex_resp(self,turn):
        value_map={}
        restored = turn['resp']
        restored=restored.replace('<sos_r>','')
        restored=restored.replace('<eos_r>','')
        restored.strip()
        restored = restored.capitalize()
        restored = restored.replace(' -s', 's')
        restored = restored.replace(' -ly', 'ly')
        restored = restored.replace(' -er', 'er')
        constraint_dict=self.reader.bspan_to_constraint_dict(turn['bspn'])#{'hotel': {'stay': '3'}, 'restaurant': {'people': '4'}}
        mat_ents = self.reader.db.get_match_num(constraint_dict, True)
        #print(mat_ents)
        #print(constraint_dict)
        if '[value_car]' in restored:
            restored = restored.replace('[value_car]', 'BMW')
            value_map['taxi']={}
            value_map['taxi']['car']='BMW'

        # restored.replace('[value_phone]', '830-430-6666')
        domain=[]
        for d in turn['turn_domain']:
            if d.startswith('['):
                domain.append(d[1:-1])
            else:
                domain.append(d)

        for d in domain:
            constraint = constraint_dict.get(d,None)
            if d not in value_map:
                value_map[d]={}
            if constraint:
                if 'stay' in constraint and '[value_stay]' in restored:
                    restored = restored.replace('[value_stay]', constraint['stay'])
                    value_map[d]['stay']=constraint['stay']
                if 'day' in constraint and '[value_day]' in restored:
                    restored = restored.replace('[value_day]', constraint['day'])
                    value_map[d]['day']=constraint['day']
                if 'people' in constraint and '[value_people]' in restored:
                    restored = restored.replace('[value_people]', constraint['people'])
                    value_map[d]['people']=constraint['people']
                if 'time' in constraint and '[value_time]' in restored:
                    restored = restored.replace('[value_time]', constraint['time'])
                    value_map[d]['time']=constraint['time']
                if 'type' in constraint and '[value_type]' in restored:
                    restored = restored.replace('[value_type]', constraint['type'])
                    value_map[d]['type']=constraint['type']
                if d in mat_ents and len(mat_ents[d])==0:
                    for s in constraint:
                        if s == 'pricerange' and d in ['hotel', 'restaurant'] and 'price]' in restored:
                            restored = restored.replace('[value_price]', constraint['pricerange'])
                            value_map[d]['price']=constraint['pricerange']
                        if s+']' in restored:
                            restored = restored.replace('[value_%s]'%s, constraint[s])
                            value_map[d][s]=constraint[s]

            if '[value_choice' in restored and mat_ents.get(d):
                restored = restored.replace('[value_choice]', str(len(mat_ents[d])))
                value_map[d]['choice']=str(len(mat_ents[d]))
        if '[value_choice' in restored:
            restored = restored.replace('[value_choice]', str(random.choice([1,2,3,4,5])))


        ent = mat_ents.get(domain[-1], [])
        d=domain[-1]
        if d not in value_map:
            value_map[d]={}
        if ent:
            # handle multiple [value_xxx] tokens first
            restored_split = restored.split()
            token_count = Counter(restored_split)
            for idx, t in enumerate(restored_split):
                if '[value' in t and token_count[t]>1 and token_count[t]<=len(ent):
                    id1=t.index('_')
                    id2=t.index(']')
                    slot = t[id1+1:id2]
                    pattern = r'\['+t[1:-1]+r'\]'
                    for e in ent:
                        if e.get(slot):
                            if domain[-1] == 'hotel' and slot == 'price':
                                slot = 'pricerange'
                            if slot in ['name', 'address']:
                                rep = ' '.join([i.capitalize() if i not in stopwords else i for i in e[slot].split()])
                            elif slot in ['id','postcode']:
                                rep = e[slot].upper()
                            else:
                                rep = e[slot]
                            restored = re.sub(pattern, rep, restored, 1)
                            value_map[d][slot]=rep
                        elif slot == 'price' and  e.get('pricerange'):
                            restored = re.sub(pattern, e['pricerange'], restored, 1)
                            value_map[d][slot]=e['pricerange']

            # handle normal 1 entity case
            ent = ent[0]
            if d=='train':
                if ent['id']=="tr2835":
                    temp=1
            ents_list=self.reader.db.dbs[domain[-1]]
            ref_no=ents_list.index(ent)
            if ref_no>9:
                if '[value_reference]' in restored:
                    restored = restored.replace('[value_reference]', '000000'+str(ref_no))
                    value_map[d]['reference']='000000'+str(ref_no)
            else:
                if '[value_reference]' in restored:
                    restored = restored.replace('[value_reference]', '0000000'+str(ref_no))
                    value_map[d]['reference']='0000000'+str(ref_no)
            for t in restored.split():
                if '[value' in t:
                    id1=t.index('_')
                    id2=t.index(']')
                    slot = t[id1+1:id2]
                    if ent.get(slot):
                        if domain[-1] == 'hotel' and slot == 'price':
                            slot = 'pricerange'
                        if slot in ['name', 'address']:
                            rep = ' '.join([i.capitalize() if i not in stopwords else i for i in ent[slot].split()])
                        elif slot in ['id','postcode']:
                            rep = ent[slot].upper()
                        else:
                            rep = ent[slot]
                        # rep = ent[slot]
                        restored = restored.replace(t, rep)
                        value_map[d][slot]=rep
                        # restored = restored.replace(t, ent[slot])
                    elif slot == 'price' and  ent.get('pricerange'):
                        restored = restored.replace(t, ent['pricerange'])
                        value_map[d][slot]=ent['pricerange']
                        # else:
                        #     print(restored, domain)


        restored = restored.replace('[value_phone]', '01223462354')
        restored = restored.replace('[value_postcode]', 'CB12DP')
        restored = restored.replace('[value_address]', 'Parkside, Cambridge')
        restored = restored.replace('[value_people]', 'several')
        restored = restored.replace('[value_day]', 'Saturday')
        restored = restored.replace('[value_time]', '12:00')
        
        #restored = restored.replace('[value_area]', 'centre')

        for t in restored.split():
            if '[value' in t:
                if '[value_name]' in t:
                    temp=1
                restored = restored.replace(t, 'UNKNOWN')

        restored = restored.split()
        for idx, w in enumerate(restored):
            if idx>0 and restored[idx-1] in ['.', '?', '!']:
                restored[idx]= restored[idx].capitalize()
        restored = ' '.join(restored)

        return restored.strip(), value_map

    def generate(self,context):
        #first generate bspn then query for db finally genrate act and response
        #input: encoded token ids
        #ouput: decoded turn
        bs_max_len=70
        act_max_len=20
        resp_max_len=60
        new_turn={}
        self.model.eval()

        with torch.no_grad():
            input_ids=context + [self.sos_b_id]
            max_len=1024-bs_max_len
            if len(input_ids)>max_len:
                input_ids=input_ids[-max_len:]
            context_length=len(input_ids)
            outputs = self.model.generate(input_ids=torch.tensor([input_ids]).to(self.model.device),
                                        pad_token_id=self.tokenizer.eos_token_id,
                                        max_length=context_length + bs_max_len, eos_token_id=self.eos_b_id)
            generated = outputs[0].cpu().numpy().tolist()
            bspn = self.tokenizer.decode(generated[context_length-1:]) #start with <sos_b>

            bspn=bspn.replace('portugese', 'portuguese')
            turn_domain=self.extract_domain(bspn)
            db_result = self.reader.bspan_to_DBpointer(bspn, turn_domain)#[db_x]
            db_result = '<sos_db> '+ db_result + ' <eos_db>'
            db = self.tokenizer.encode(db_result)#token ids

            input_ids=generated + db + [self.sos_a_id]
            max_len=1024-act_max_len
            if len(input_ids)>max_len:
                input_ids=input_ids[-max_len:]
            context_length=len(input_ids)
            outputs = self.model.generate(input_ids=torch.tensor([input_ids]).to(self.model.device),
                                        pad_token_id=self.tokenizer.eos_token_id,
                                        max_length=context_length + act_max_len, eos_token_id=self.eos_a_id)
            generated = outputs[0].cpu().numpy().tolist()
            aspn = self.tokenizer.decode(generated[context_length-1:])
            if cfg.sys_act_ctrl:
                aspn_fix= '<sos_a> ' + self.fix_aspn(aspn) + ' <eos_a>'
                generated = generated[:context_length-1] + self.reader.modified_encode(aspn_fix)

            input_ids=generated + [self.sos_r_id]
            max_len=1024-resp_max_len
            if len(input_ids)>max_len:
                input_ids=input_ids[-max_len:]
            context_length=len(input_ids)
            outputs = self.model.generate(input_ids=torch.tensor([input_ids]).to(self.model.device),
                                        pad_token_id=self.tokenizer.eos_token_id,
                                        max_length=context_length + resp_max_len, eos_token_id=self.eos_r_id)
            generated = outputs[0].cpu().numpy().tolist()
            resp = self.tokenizer.decode(generated[context_length-1:])

        new_turn['user']='<sos_u> '+ self.user +' <eos_u>'
        new_turn['bspn'], new_turn['db'], new_turn['aspn'], new_turn['resp'], new_turn['turn_domain']=\
            bspn, db_result, aspn, resp, turn_domain

        return new_turn

    def extract_domain(self,bspn):
        constraint_dict = self.reader.bspan_to_constraint_dict(bspn)#{'hotel': {'stay': '3'}, 'restaurant': {'people': '4'}}
        turn_dom_bs = []
        for domain, info_slots in constraint_dict.items():
            if info_slots:
                if domain not in self.constraint_dict:
                    turn_dom_bs.append(domain)
                elif self.constraint_dict[domain] != constraint_dict[domain]:
                    turn_dom_bs.append(domain)

        # get turn domain
        turn_domain = turn_dom_bs
        if not turn_domain:
            turn_domain = self.turn_domain
        if len(turn_domain) == 2 and 'general' in turn_domain:
            turn_domain.remove('general')
        if len(turn_domain) == 2:
            if len(self.turn_domain) == 1 and self.turn_domain[0] == turn_domain[1]:
                turn_domain = turn_domain[::-1]
        for domain in all_domains:
            if domain in self.user:
                turn_domain = [domain]
                break
        self.constraint_dict = copy.deepcopy(constraint_dict)
        self.turn_domain = copy.deepcopy(turn_domain)
        return turn_domain
    
    def fix_aspn(self, aspn):
        act_dict=self.reader.aspan_to_act_dict(aspn)
        act_dict, reward=fix_act(act_dict)
        aspn=self.reader.act_dict_to_aspan(act_dict)
        return aspn

    def aspn_to_action(self, aspn, value_map):
        act_list=self.reader.aspan_to_act_list(aspn)#[domain-intent-slot]
        action_list=[]
        for act in act_list:
            action=[]
            dis=act.split('-')
            domain=dis[0] if dis[0]=='general' else dis[0].capitalize()
            intent=intent_map[dis[1]]
            if dis[2] in special_slot_map:
                slot=special_slot_map[dis[2]]
            else:
                slot=dis[2].capitalize()
            if domain=='Attraction' and slot=='Price':
                slot='Fee'
            if domain=='Train' and slot=='Price':
                slot='Ticket'
            if dis[1] in ['inform','recommend','offerbooked', 'nooffer', 'nobook'] and dis[0] in value_map:
                value=value_map[dis[0]].get(dis[2], 'none')
            elif dis[1] in ['request']:
                value='?'
            else:
                value='none'

            if slot=='None':
                slot='none'
            if value=='None':
                value='none'

            if domain in ['Hotel', 'Attraction', 'Restaurant','Taxi','Hospital']:
                if intent=='OfferBooked':
                    action=["Book","Booking", slot, value]
                elif intent=='OfferBook':
                    action=['Inform','Booking','none','none']
                elif intent=='Request' and slot not in convlab2_dict[domain]['Request']:
                    action=['Request','Booking', slot, value]
                elif domain in ['Hotel', 'Attraction', 'Restaurant'] and intent=='NoBook':
                    action=["NoBook", "Booking", slot, value]
                else:
                    action=[intent,domain,slot,value]
            else:
                action=[intent,domain,slot,value]
            action_list.append(action)

        return action_list

class turn_level_sys(Agent, Session):

    def __init__(self, path, **kwargs):
        Agent.__init__(self, name='turn-level-sys')
        Session.__init__(self, path, **kwargs)
        self.init_session()

    def response(self, user):
        user='<sos_u> '+user+' <eos_u>'
        if self.pv_bspn is None: # first turn
            bspn, db, aspn, resp = self.get_sys_response(user)
        else:
            bspn, db, aspn, resp = self.get_sys_response(user, self.pv_bspn, self.pv_resp)
        self.pv_bspn=bspn
        self.pv_resp=resp
        resp1=self.lex_resp(resp, bspn, self.turn_domain)
        turn={'user':user, 'bspn':bspn, 'db':db, 'aspn':aspn, 'resp':resp, 'lex_resp': resp1}
        self.dialog.append(turn)
        return resp1

    def lex_resp(self, resp, bspn, turn_domain):
        value_map={}
        restored = resp
        restored=restored.replace('<sos_r>','')
        restored=restored.replace('<eos_r>','')
        restored.strip()
        restored = restored.capitalize()
        restored = restored.replace(' -s', 's')
        restored = restored.replace(' -ly', 'ly')
        restored = restored.replace(' -er', 'er')
        constraint_dict=self.reader.bspan_to_constraint_dict(bspn)#{'hotel': {'stay': '3'}, 'restaurant': {'people': '4'}}
        mat_ents = self.reader.db.get_match_num(constraint_dict, True)
        #print(mat_ents)
        #print(constraint_dict)
        if '[value_car]' in restored:
            restored = restored.replace('[value_car]', 'BMW')
            value_map['taxi']={}
            value_map['taxi']['car']='BMW'

        # restored.replace('[value_phone]', '830-430-6666')
        domain=[]
        for d in turn_domain:
            if d.startswith('['):
                domain.append(d[1:-1])
            else:
                domain.append(d)

        for d in domain:
            constraint = constraint_dict.get(d,None)
            if d not in value_map:
                value_map[d]={}
            if constraint:
                if 'stay' in constraint and '[value_stay]' in restored:
                    restored = restored.replace('[value_stay]', constraint['stay'])
                    value_map[d]['stay']=constraint['stay']
                if 'day' in constraint and '[value_day]' in restored:
                    restored = restored.replace('[value_day]', constraint['day'])
                    value_map[d]['day']=constraint['day']
                if 'people' in constraint and '[value_people]' in restored:
                    restored = restored.replace('[value_people]', constraint['people'])
                    value_map[d]['people']=constraint['people']
                if 'time' in constraint and '[value_time]' in restored:
                    restored = restored.replace('[value_time]', constraint['time'])
                    value_map[d]['time']=constraint['time']
                if 'type' in constraint and '[value_type]' in restored:
                    restored = restored.replace('[value_type]', constraint['type'])
                    value_map[d]['type']=constraint['type']
                if d in mat_ents and len(mat_ents[d])==0:
                    for s in constraint:
                        if s == 'pricerange' and d in ['hotel', 'restaurant'] and 'price]' in restored:
                            restored = restored.replace('[value_price]', constraint['pricerange'])
                            value_map[d]['price']=constraint['pricerange']
                        if s+']' in restored:
                            restored = restored.replace('[value_%s]'%s, constraint[s])
                            value_map[d][s]=constraint[s]

            if '[value_choice' in restored and mat_ents.get(d):
                restored = restored.replace('[value_choice]', str(len(mat_ents[d])))
                value_map[d]['choice']=str(len(mat_ents[d]))
        if '[value_choice' in restored:
            restored = restored.replace('[value_choice]', str(random.choice([1,2,3,4,5])))


        ent = mat_ents.get(domain[-1], [])
        d=domain[-1]
        if d not in value_map:
            value_map[d]={}
        if ent:
            # handle multiple [value_xxx] tokens first
            restored_split = restored.split()
            token_count = Counter(restored_split)
            for idx, t in enumerate(restored_split):
                if '[value' in t and token_count[t]>1 and token_count[t]<=len(ent):
                    id1=t.index('_')
                    id2=t.index(']')
                    slot = t[id1+1:id2]
                    pattern = r'\['+t[1:-1]+r'\]'
                    for e in ent:
                        if e.get(slot):
                            if domain[-1] == 'hotel' and slot == 'price':
                                slot = 'pricerange'
                            if slot in ['name', 'address']:
                                rep = ' '.join([i.capitalize() if i not in stopwords else i for i in e[slot].split()])
                            elif slot in ['id','postcode']:
                                rep = e[slot].upper()
                            else:
                                rep = e[slot]
                            restored = re.sub(pattern, rep, restored, 1)
                            value_map[d][slot]=rep
                        elif slot == 'price' and  e.get('pricerange'):
                            restored = re.sub(pattern, e['pricerange'], restored, 1)
                            value_map[d][slot]=e['pricerange']

            # handle normal 1 entity case
            ent = ent[0]
            if d=='train':
                if ent['id']=="tr2835":
                    temp=1
            ents_list=self.reader.db.dbs[domain[-1]]
            ref_no=ents_list.index(ent)
            if ref_no>9:
                if '[value_reference]' in restored:
                    restored = restored.replace('[value_reference]', '000000'+str(ref_no))
                    value_map[d]['reference']='000000'+str(ref_no)
            else:
                if '[value_reference]' in restored:
                    restored = restored.replace('[value_reference]', '0000000'+str(ref_no))
                    value_map[d]['reference']='0000000'+str(ref_no)
            for t in restored.split():
                if '[value' in t:
                    id1=t.index('_')
                    id2=t.index(']')
                    slot = t[id1+1:id2]
                    if ent.get(slot):
                        if domain[-1] == 'hotel' and slot == 'price':
                            slot = 'pricerange'
                        if slot in ['name', 'address']:
                            rep = ' '.join([i.capitalize() if i not in stopwords else i for i in ent[slot].split()])
                        elif slot in ['id','postcode']:
                            rep = ent[slot].upper()
                        else:
                            rep = ent[slot]
                        # rep = ent[slot]
                        restored = restored.replace(t, rep)
                        value_map[d][slot]=rep
                        # restored = restored.replace(t, ent[slot])
                    elif slot == 'price' and  ent.get('pricerange'):
                        restored = restored.replace(t, ent['pricerange'])
                        value_map[d][slot]=ent['pricerange']
                        # else:
                        #     print(restored, domain)       
        #restored = restored.replace('[value_area]', 'centre')
        for t in restored.split():
            if '[value' in t:
                slot=t[7:-1]
                value='UNKNOWN'
                for domain, sv in constraint_dict.items():
                    if isinstance(sv, dict) and slot in sv:
                        value=sv[slot]
                        break
                if value=='UNKNOWN':
                    for domain in mat_ents:
                        ent=mat_ents[domain][0]
                        if slot in ent:
                            if slot in ['name', 'address']:
                                value=' '.join([i.capitalize() if i not in stopwords else i for i in ent[slot].split()])
                            elif slot in ['id', 'postcode']:
                                value=ent[slot].upper()
                            else:
                                value=ent[slot]
                            break          
                restored = restored.replace(t, value)
        restored = restored.replace('[value_phone]', '01223462354')
        restored = restored.replace('[value_postcode]', 'CB12DP')
        restored = restored.replace('[value_address]', 'Parkside, Cambridge')
        restored = restored.replace('[value_people]', 'several')
        restored = restored.replace('[value_day]', 'Saturday')
        restored = restored.replace('[value_time]', '12:00')
        restored = restored.split()
        for idx, w in enumerate(restored):
            if idx>0 and restored[idx-1] in ['.', '?', '!']:
                restored[idx]= restored[idx].capitalize()
        restored = ' '.join(restored)

        return restored.strip()
    
    def init_session(self):
        self.pv_bspn=None
        self.pv_resp=None
        self.dialog=[]

if __name__ == '__main__':
    s = LsGPT()
    bspn="[hotel] pricerange cheap type hotel"
    print(s.reader.bspan_to_constraint_dict(bspn))
    '''
    aspn='<sos_a> [hotel] [inform] price name type area stars phone [offerbook] <eos_a>'
    act_dict=s.reader.aspan_to_act_dict(aspn)
    print(act_dict)
    print(fix_act(act_dict))
    print(s.reader.act_dict_to_aspan(fix_act(act_dict)))
    '''
    #print(s.response("I want to find a cheap restaurant"))
    #print(s.response("ok, what is the address ?"))