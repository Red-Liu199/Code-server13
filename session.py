import torch
import random
import time
import numpy as np
import ontology
import json
from config import global_config as cfg
from reader import MultiWozReader
from utils import modified_encode
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from prepare_us_data import goal_to_gpan
class turn_level_session(object):
    # turn-level DS: prev_BS + prev_Resp + User_utter + BS + DB + Sys_act + Resp
    # turn-level US: Goal + prev_Resp + User_act +User_utter
    def __init__(self, DS_path, US_path):
        self.DS=GPT2LMHeadModel.from_pretrained(DS_path)
        self.US=GPT2LMHeadModel.from_pretrained(US_path)
        self.DS.to(cfg.DS_device)
        self.US.to(cfg.US_device)
        self.DS_tok=GPT2Tokenizer.from_pretrained(DS_path)
        self.US_tok=GPT2Tokenizer.from_pretrained(US_path)
        self.reader = MultiWozReader(self.DS_tok)
        self.get_special_ids()
        self.end_tokens=set(['[general]', '[bye]', '[welcome]', '[thank]','[greet]', 
            '[reqmore]', '<sos_a>', '<eos_a>', '<sos_ua>', '<eos_ua>'])
    
    def get_special_ids(self):

        self.sos_ua_id=self.US_tok.convert_tokens_to_ids('<sos_ua>')
        self.eos_ua_id=self.US_tok.convert_tokens_to_ids('<eos_ua>')
        self.sos_u_id=self.US_tok.convert_tokens_to_ids('<sos_u>')
        self.eos_u_id=self.US_tok.convert_tokens_to_ids('<eos_u>')

        self.sos_b_id=self.DS_tok.convert_tokens_to_ids('<sos_b>')
        self.eos_b_id=self.DS_tok.convert_tokens_to_ids('<eos_b>')
        self.sos_a_id=self.DS_tok.convert_tokens_to_ids('<sos_a>')
        self.eos_a_id=self.DS_tok.convert_tokens_to_ids('<eos_a>')
        self.sos_r_id=self.DS_tok.convert_tokens_to_ids('<sos_r>')
        self.eos_r_id=self.DS_tok.convert_tokens_to_ids('<eos_r>')
        

    def interact(self, goal=None):
        # Initialization and restrictions
        max_turns=20
        gen_dial=[]
        # If goal is None, sample a goal from training set
        if goal is None:
            dial_id=random.sample(self.reader.train_list, 1)[0]
            goal=self.reader.data[dial_id]['goal']
        gpan=goal_to_gpan(goal)
        self.turn_domain=''
        for i in range(max_turns):
            turn={}
            if i==0:
                user_act, user = self.get_user_utterance(gpan, pv_resp='<sos_r> <eos_r>')
                bspn, db, aspn, resp = self.get_sys_response(user)
            else:
                user_act, user = self.get_user_utterance(gpan, pv_resp=pv_resp)
                bspn, db, aspn, resp = self.get_sys_response(user, pv_bspan, pv_resp)
            
            turn['gpan'], turn['usr_act'], turn['user'], turn['bspn'], turn['db'], \
                turn['aspn'], turn['resp'] = gpan, user_act, user, bspn, db, aspn, resp
            gen_dial.append(turn)
            pv_resp=resp
            pv_bspan=bspn
            if set(user_act.split()).issubset(self.end_tokens) and set(aspn.split()).issubset(self.end_tokens):
                break
        return gen_dial

    def get_sys_response(self, user_utter, pv_b=None, pv_resp=None):
        # First generate bspn then query for db finally genrate act and response
        bs_max_len=60
        act_max_len=20
        resp_max_len=60
        self.DS.eval()

        with torch.no_grad():
            if pv_resp is None: # first turn
                input_ids=self.reader.modified_encode(user_utter) + [self.sos_b_id]
            else:
                input_ids=self.reader.modified_encode(pv_b+pv_resp+user_utter) + [self.sos_b_id]
            max_len=1024-bs_max_len
            if len(input_ids)>max_len:
                input_ids=input_ids[-max_len:]
            context_length=len(input_ids)
            outputs = self.DS.generate(input_ids=torch.tensor([input_ids]).to(self.DS.device),
                                        pad_token_id=self.DS_tok.eos_token_id,
                                        max_length=context_length + bs_max_len, eos_token_id=self.eos_b_id)
            generated = outputs[0].cpu().numpy().tolist()
            bspn = self.DS_tok.decode(generated[context_length-1:]) #start with <sos_b>
            cons=self.reader.bspan_to_constraint_dict(bspn)
            cur_domain=list(cons.keys())
            if cur_domain==[]:
                db_result = '<sos_db> '+ '[db_0]' + ' <eos_db>'
                db = self.DS_tok.encode(db_result)#token ids
            else:
                if len(cur_domain)==1:
                    self.turn_domain=cur_domain
                else:
                    if pv_b is None: # In rare cases, there are more than one domain in the first turn
                        max_slot_num=0 # We choose the domain with most slots as the current domain
                        for domain in cur_domain:
                            if len(cons[domain])>max_slot_num:
                                self.turn_domain=[domain]
                                max_slot_num=len(cons[domain])
                    else:
                        pv_domain=list(self.reader.bspan_to_constraint_dict(pv_b).keys())
                        for domain in cur_domain:
                            if domain not in pv_domain: # new domain
                                # if domains are all the same, self.domain will not change
                                self.turn_domain=[domain]

                #bspn=bspn.replace('portugese', 'portuguese')
                db_result = self.reader.bspan_to_DBpointer(bspn, self.turn_domain) #[db_x]
                db_result = '<sos_db> '+ db_result + ' <eos_db>'
                db = self.DS_tok.encode(db_result)#token ids

            input_ids=generated + db + [self.sos_a_id]
            max_len=1024-act_max_len
            if len(input_ids)>max_len:
                input_ids=input_ids[-max_len:]
            context_length=len(input_ids)
            outputs = self.DS.generate(input_ids=torch.tensor([input_ids]).to(self.DS.device),
                                        pad_token_id=self.DS_tok.eos_token_id,
                                        max_length=context_length + act_max_len, eos_token_id=self.eos_a_id)
            generated = outputs[0].cpu().numpy().tolist()
            aspn = self.DS_tok.decode(generated[context_length-1:])

            input_ids=generated + [self.sos_r_id]
            max_len=1024-resp_max_len
            if len(input_ids)>max_len:
                input_ids=input_ids[-max_len:]
            context_length=len(input_ids)
            outputs = self.DS.generate(input_ids=torch.tensor([input_ids]).to(self.DS.device),
                                        pad_token_id=self.DS_tok.eos_token_id,
                                        max_length=context_length + resp_max_len, eos_token_id=self.eos_r_id)
            generated = outputs[0].cpu().numpy().tolist()
            resp = self.DS_tok.decode(generated[context_length-1:])

        return bspn, db_result, aspn, resp

    def get_user_utterance(self, gpan, pv_resp):
        # First generate user act then user utterance
        act_max_len=25
        utter_max_len=55
        self.US.eval()

        with torch.no_grad():
            input_ids=modified_encode(self.US_tok,gpan+pv_resp) + [self.sos_ua_id]
            max_len=1024-act_max_len
            if len(input_ids)>max_len:
                input_ids=input_ids[-max_len:]
            context_length=len(input_ids)
            outputs = self.US.generate(input_ids=torch.tensor([input_ids]).to(self.US.device),
                                        pad_token_id=self.US_tok.eos_token_id,
                                        max_length=context_length + act_max_len, eos_token_id=self.eos_ua_id)
            generated = outputs[0].cpu().numpy().tolist()
            user_act = self.US_tok.decode(generated[context_length-1:]) #start with <sos_ua>

            input_ids=generated + [self.sos_u_id]
            max_len=1024-utter_max_len
            if len(input_ids)>max_len:
                input_ids=input_ids[-max_len:]
            context_length=len(input_ids)
            outputs = self.US.generate(input_ids=torch.tensor([input_ids]).to(self.US.device),
                                        pad_token_id=self.US_tok.eos_token_id,
                                        max_length=context_length + utter_max_len, eos_token_id=self.eos_u_id)
            generated = outputs[0].cpu().numpy().tolist()
            user = self.US_tok.decode(generated[context_length-1:])


        return user_act, user

    def interact_by_batch(self, batch_size=cfg.batch_size):
        max_turns=20
        gen_batch=[[] for _ in range(batch_size)]
        end_batch=[0 for _ in range(batch_size)]
        gpan_batch=[]
        bs_max_len=50
        act_max_len=20
        resp_max_len=60
        dial_id_batch=random.sample(self.reader.train_list, batch_size)
        for dial_id in dial_id_batch:
            goal=self.reader.data[dial_id]['goal']
            gpan=goal_to_gpan(goal)
            gpan_batch.append(gpan)
        self.turn_domain_batch=['' for i in range(batch_size)]
        pv_resp_batch=None
        pv_bspn_batch=None
        for i in range(max_turns):
            # generate user act batch
            contexts=self.get_us_contexts(gpan_batch, pv_resp_batch)
            contexts_ids=self.convert_batch_tokens_to_ids(self.US_tok, contexts)
            user_act_batch_ids=self.generate_batch(self.US, contexts_ids, act_max_len, self.eos_ua_id)
            user_act_batch=self.convert_batch_ids_to_tokens(self.US_tok, user_act_batch_ids, 
                self.sos_ua_id, self.eos_ua_id)
            # generate user batch
            contexts=self.get_us_contexts(gpan_batch, pv_resp_batch, user_act_batch)
            contexts_ids=self.convert_batch_tokens_to_ids(self.US_tok, contexts)
            user_batch_ids=self.generate_batch(self.US, contexts_ids, resp_max_len, self.eos_u_id)
            user_batch=self.convert_batch_ids_to_tokens(self.US_tok, user_batch_ids, 
                self.sos_u_id, self.eos_u_id)
            # generate bspn batch
            contexts=self.get_ds_contexts(user_batch, pv_bspn_batch, pv_resp_batch)
            contexts_ids=self.convert_batch_tokens_to_ids(self.DS_tok, contexts)
            bspn_batch_ids=self.generate_batch(self.DS, contexts_ids, bs_max_len, self.eos_b_id)
            bspn_batch=self.convert_batch_ids_to_tokens(self.DS_tok, bspn_batch_ids, 
                self.sos_b_id, self.eos_b_id)
            db_batch=self.get_db_batch(bspn_batch, pv_bspn_batch)
            # generate act batch
            contexts=self.get_ds_contexts(user_batch, pv_bspn_batch, pv_resp_batch, bspn_batch, db_batch)
            contexts_ids=self.convert_batch_tokens_to_ids(self.DS_tok, contexts)
            aspn_batch_ids=self.generate_batch(self.DS, contexts_ids, act_max_len, self.eos_a_id)
            aspn_batch=self.convert_batch_ids_to_tokens(self.DS_tok, aspn_batch_ids, 
                self.sos_a_id, self.eos_a_id)
            # generate resp batch
            contexts=self.get_ds_contexts(user_batch, pv_bspn_batch, pv_resp_batch, bspn_batch, db_batch,aspn_batch)
            contexts_ids=self.convert_batch_tokens_to_ids(self.DS_tok, contexts)
            resp_batch_ids=self.generate_batch(self.DS, contexts_ids, resp_max_len, self.eos_r_id)
            resp_batch=self.convert_batch_ids_to_tokens(self.DS_tok, resp_batch_ids, 
                self.sos_r_id, self.eos_r_id)
            
            # before next turn
            pv_bspn_batch=bspn_batch
            pv_resp_batch=resp_batch

            # collect dialogs and judge stop
            for batch_id in range(batch_size):
                user_act=user_act_batch[batch_id]
                aspn=aspn_batch[batch_id]
                if not end_batch[batch_id]:
                    turn={}
                    turn['gpan']=gpan_batch[batch_id]
                    turn['usr_act']=user_act_batch[batch_id]
                    turn['user']=user_batch[batch_id]
                    turn['bspn']=bspn_batch[batch_id]
                    turn['db']=db_batch[batch_id]
                    turn['aspn']=aspn_batch[batch_id]
                    turn['resp']=resp_batch[batch_id]
                    gen_batch[batch_id].append(turn)
                if set(user_act.split()).issubset(self.end_tokens) and set(aspn.split()).issubset(self.end_tokens):
                    end_batch[batch_id]=1
            if all(end_batch):
                break
        return gen_batch
    
    def generate_batch(self, model, contexts, max_len, eos_id):
        # generate by batch
        # contexts: a batch of ids
        # max_len: the max generated length
        # eos_id: the end id
        # return: a batch of ids
        batch_size=len(contexts)
        end_flag=np.zeros(batch_size)
        past_key_values=None
        inputs,attentions=self.reader.batch_align(contexts,left_len=max_len,return_attn=True)
        inputs=torch.tensor(inputs).to(model.device)
        attentions=torch.tensor(attentions).to(model.device)
        model.eval()
        with torch.no_grad():
            for i in range(max_len):
                position_ids = attentions.long().cumsum(-1) - 1
                position_ids.masked_fill_(attentions == 0, 1)
                if past_key_values is not None:
                    position_ids=position_ids[:, -1].unsqueeze(-1)
                outputs=model(inputs,attention_mask=attentions,position_ids=position_ids,\
                        return_dict=True,use_cache=True,past_key_values=past_key_values)

                past_key_values=outputs.past_key_values

                preds=outputs.logits[:,-1,:].argmax(-1)#B
                if i==0:
                    gen_tensor=preds.unsqueeze(1)
                else:
                    gen_tensor=torch.cat([gen_tensor,preds.unsqueeze(1)],dim=1)
                attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(model.device)),dim=1)
                inputs=preds.unsqueeze(1)
                end_flag+=(preds.cpu().numpy()==eos_id).astype(float)
                if sum(end_flag==0)==0:
                    break
               
        return gen_tensor.cpu().tolist()
    
    def get_us_contexts(self, gpan_batch, pv_resp_batch=None, user_act_batch=None):
        contexts=[]
        if pv_resp_batch==None:# first turn
            if user_act_batch is None:
                for gpan in gpan_batch:
                    context = gpan + '<sos_r> <eos_r>' + '<sos_ua>'
                    contexts.append(context)
            else:
                for gpan, ua in zip(gpan_batch, user_act_batch):
                    context = gpan + '<sos_r> <eos_r>' + ua + '<sos_u>'
                    contexts.append(context)
        else:
            if user_act_batch is None:
                for gpan, pv_r in zip(gpan_batch, pv_resp_batch):
                    context = gpan + pv_r + '<sos_ua>'
                    contexts.append(context)
            else:
                for gpan, pv_r, ua in zip(gpan_batch, pv_resp_batch, user_act_batch):
                    context = gpan + pv_r + ua + '<sos_u>'
                    contexts.append(context)
        return contexts
    
    def get_ds_contexts(self, user_batch, pv_bspn_batch=None, pv_resp_batch=None, bspn_batch=None, 
        db_batch=None, aspn_batch=None):
        contexts=[]
        if pv_resp_batch is None: # first turn
            if bspn_batch is None:
                for u in user_batch:
                    contexts.append(u + '<sos_b>')
            elif aspn_batch is None:
                for u, b in zip(user_batch, bspn_batch):
                    contexts.append(u + b + '<sos_a>')
            else:
                for u, b, db, a in zip(user_batch, bspn_batch, db_batch, aspn_batch):
                    contexts.append(u + b + db + a + '<sos_r>')
        else:
            if bspn_batch is None:
                for pv_b, pv_r, u in zip(pv_bspn_batch, pv_resp_batch, user_batch):
                    contexts.append(pv_b + pv_r + u + '<sos_b>')
            elif aspn_batch is None:
                for pv_b, pv_r, u, b in zip(pv_bspn_batch, pv_resp_batch, user_batch, bspn_batch):
                    contexts.append(pv_b + pv_r + u + b + '<sos_a>')
            else:
                for pv_b, pv_r, u, b, db, a in zip(pv_bspn_batch, pv_resp_batch, user_batch, bspn_batch, db_batch, aspn_batch):
                    contexts.append(pv_b + pv_r + u + b + db + a + '<sos_r>')
        return contexts

    def get_db_batch(self, bs_batch, pv_bs_batch=None):

        db_batch=[]
        for i, bspn in enumerate(bs_batch):
            cons=self.reader.bspan_to_constraint_dict(bspn)
            cur_domain=list(cons.keys())
            if cur_domain==[]:
                db_result='<sos_db> '+ '[db_0]' + ' <eos_db>'
            else:
                if len(cur_domain)==1:
                    self.turn_domain_batch[i]=cur_domain
                else:
                    if pv_bs_batch is None:
                        max_slot_num=0 # We choose the domain with most slots as the current domain
                        for domain in cur_domain:
                            if len(cons[domain])>max_slot_num:
                                self.turn_domain_batch[i]=[domain]
                                max_slot_num=len(cons[domain])
                    else:
                        pv_domain=list(self.reader.bspan_to_constraint_dict(pv_bs_batch[i]).keys())
                        for domain in cur_domain:
                            if domain not in pv_domain: # new domain
                                # if domains are all the same, self.domain will not change
                                self.turn_domain_batch[i]=[domain]

                #bspn=bspn.replace('portugese', 'portuguese')
                db_result = self.reader.bspan_to_DBpointer(bspn, self.turn_domain_batch[i]) #[db_x]
                db_result = '<sos_db> '+ db_result + ' <eos_db>'
            db_batch.append(db_result)

        return db_batch

    def convert_batch_ids_to_tokens(self, tokenizer, input_ids, sos_id, eos_id):
        # input_ids: B*T
        # output: B*string
        outputs=[]
        for sent_ids in input_ids:
            if eos_id in sent_ids:
                sent_ids=sent_ids[:sent_ids.index(eos_id)+1]
            else:
                sent_ids[-1]=eos_id
            sent_ids=[sos_id]+sent_ids
            outputs.append(tokenizer.decode(sent_ids))
        return outputs

    def convert_batch_tokens_to_ids(self, tokenizer, contexts):
        outputs=[]
        for context in contexts:
            outputs.append(modified_encode(tokenizer, context))
        return outputs


if __name__=='__main__':
    DS_path='experiments_21/turn-level-DS/best_score_model'
    US_path='experiments_21/turn-level-US/best_score_model'
    save_path='example.json'
    dials1=[]
    dials2=[]
    session=turn_level_session(DS_path, US_path)
    dial_num=16
    st=time.time()
    for i in range(dial_num):
        gen_dial=session.interact()
        print(len(gen_dial))
        dials1.append(gen_dial)
    print('Time consuming of one by one interaction: {:.2f} s'.format(time.time()-st))
    json.dump(dials1, open('example1.json','w'),indent=2)
    st=time.time()
    gen_batch=session.interact_by_batch(dial_num)
    print([len(dial) for dial in gen_batch])
    print('Time consuming of batch interaction: {:.2f} s'.format(time.time()-st))
    dials2=gen_batch
    json.dump(dials2, open('example2.json', 'w'), indent=2)