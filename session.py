import torch
import random
import time
import os
import logging
import argparse
import numpy as np
import ontology
import json
import shutil
import torch.nn as nn
from config import global_config as cfg
from reader import MultiWozReader
from eval import MultiWozEvaluator
from utils import modified_encode
from torch.utils.tensorboard import SummaryWriter
from train_semi import parse_arg_cfg
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

class turn_level_session(object):
    # turn-level DS: prev_BS + prev_Resp + User_utter + BS + DB + Sys_act + Resp
    # turn-level US: Goal + prev_Resp + User_act +User_utter
    def __init__(self, DS_path, US_path=None, device1='cpu', device2='cpu'):
        self.DS=GPT2LMHeadModel.from_pretrained(DS_path)
        self.DS.to(device1)
        self.DS_tok=GPT2Tokenizer.from_pretrained(DS_path)
        if US_path:
            self.US=GPT2LMHeadModel.from_pretrained(US_path)
            self.US.to(device2)
            self.US_tok=GPT2Tokenizer.from_pretrained(US_path)
        else:
            print('No user simulator for the turn-level session')
        self.reader = MultiWozReader(self.DS_tok)
        self.evaluator=MultiWozEvaluator(self.reader)
        self.get_special_ids()
        self.end_tokens=set(['[general]', '[bye]', '[welcome]', '[thank]', '[reqmore]', '<sos_a>', '<eos_a>', '<sos_ua>', '<eos_ua>'])
        self.global_output=2
        # tensorboard
        if cfg.save_log:
            log_path='./log_rl/log_{}'.format(cfg.exp_no)
            if os.path.exists(log_path):
                shutil.rmtree(log_path)
                os.mkdir(log_path)
            else:
                os.mkdir(log_path)
            self.tb_writer = SummaryWriter(log_dir=log_path)
        else:
            self.tb_writer = None

    def get_special_ids(self):
        if hasattr(self, 'US_tok'):
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

    def interact(self, goal=None, return_reward=False):
        # Initialization and restrictions
        max_turns=20
        gen_dial=[]
        # If goal is None, sample a goal from training set
        if goal is None:
            dial_id=random.sample(self.reader.train_list, 1)[0]
            init_goal=self.reader.data[dial_id]['goal']
            goal=self.reader.goal_norm(init_goal)
        self.turn_domain=''
        self.goal_list=[]
        for i in range(max_turns):
            turn={}
            if i==0:
                gpan='<sos_g> '+self.reader.goal_to_gpan(goal)+' <eos_g>'
                user_act, user = self.get_user_utterance(gpan, pv_resp='<sos_r> <eos_r>')
                bspn, db, aspn, resp = self.get_sys_response(user)
            else:
                pv_constraint=self.reader.bspan_to_constraint_dict(pv_bspan)
                pv_user_act_dict=self.reader.aspan_to_act_dict(pv_user_act, side='user')
                goal=self.reader.update_goal(goal,pv_user_act_dict,pv_constraint)# update the goal
                gpan='<sos_g> '+self.reader.goal_to_gpan(goal)+' <eos_g>'
                user_act, user = self.get_user_utterance(gpan, pv_resp=pv_resp)
                bspn, db, aspn, resp = self.get_sys_response(user, pv_bspan, pv_resp)
            self.goal_list.append(goal)
            turn['gpan'], turn['usr_act'], turn['user'], turn['bspn'], turn['db'], \
                turn['aspn'], turn['resp'] = gpan, user_act, user, bspn, db, aspn, resp
            gen_dial.append(turn)
            pv_resp=resp
            pv_bspan=bspn
            pv_user_act=user_act
            if (set(user_act.split()).issubset(self.end_tokens) and set(aspn.split()).issubset(self.end_tokens)) or goal=={}:
                break
        US_rewards=self.get_US_reward(gen_dial, self.goal_list)
        DS_rewards=self.get_DS_reward(gen_dial, init_goal)
        success, match=self.get_metrics(init_goal,  gen_dial)
        #print('US rewards:', US_rewards)
        #print('DS rewards:', DS_rewards)
        #print('Success:', success, 'Match:', inform)
        if cfg.return_reward:
            return gen_dial, US_rewards, DS_rewards, success, match
        return gen_dial

    
    def run_RL(self):
        json.dump(cfg.__dict__,open(os.path.join(cfg.exp_path,'cfg_all.json'),'w'),indent=2)
        cfg.origin_batch_size=cfg.batch_size
        cfg.batch_size=cfg.batch_size*cfg.gradient_accumulation_steps
        # The total turn samples per epoch: dial_per_epoch*avg_turn_num, we think avg_turn_num=8
        self.optimizer, self.scheduler=self.get_optimizers(num_dials=cfg.rl_dial_per_epoch*8, model=self.DS)
        if cfg.joint_train:
            self.optimizer_us, self.scheduler_us=self.get_optimizers(num_dials=cfg.rl_dial_per_epoch*8, model=self.US)
        # sample some dialogs for validation
        dial_id_batches=[]
        for _ in range(4): # total dialogs 4*interaction_batch_size
            dial_id_batches.append(random.sample(self.reader.train_list + self.reader.dev_list, cfg.interaction_batch_size))
        if cfg.init_eval:
            logging.info('Initial validation')
            self.validate(dial_id_batches)
        
        max_score=0
        early_stop_count=cfg.early_stop_count
        weight_decay_count=cfg.weight_decay_count
        lr=cfg.lr
        logging.info('Dialogs per rl epoch:{}'.format(cfg.rl_dial_per_epoch))
        self.DS_training_steps=0
        self.US_training_steps=0
        for epoch in range(cfg.epoch_num):
            st=time.time()
            avg_DS_reward, avg_US_reward=self.run_RL_epoch()
            logging.info('Epoch:{}, time:{:.3f} min, DS steps:{}, US steps:{}'.format(
                epoch, (time.time()-st)/60, self.DS_training_steps, self.US_training_steps))
            logging.info('Training -- Avg DS reward:{:3f}, avg US reward:{:3f}'.format(avg_DS_reward, avg_US_reward))
            
            DS_reward, US_reward, avg_turn, success, match =self.validate(dial_id_batches)
            eval_metric=success+match

            if eval_metric>max_score:
                max_score=eval_metric
                self.save_model()
                logging.info('model saved in {}'.format(cfg.exp_path))
                early_stop_count=cfg.early_stop_count
            else:
                early_stop_count-=1
                weight_decay_count-=1
            if early_stop_count==0 and cfg.early_stop:
                print('early stop')
                break
            if weight_decay_count==0 and not cfg.use_scheduler:
                for group in self.optimizer.param_groups:
                    group['lr'] = group['lr']*cfg.lr_decay
                if cfg.joint_train:
                    for group in self.optimizer_us.param_groups:
                        group['lr'] = group['lr']*cfg.lr_decay
                print("learning rate decay to {}".format(lr))
                weight_decay_count = cfg.weight_decay_count
                if lr<1e-9:
                    print('learning rate too small, break')
                    break

            if self.tb_writer:
                self.tb_writer.add_scalar('Train_DS_reward', avg_DS_reward, epoch)
                self.tb_writer.add_scalar('Train_US_reward', avg_US_reward, epoch)        
                self.tb_writer.add_scalar('Dev_DS_reward', DS_reward, epoch)
                self.tb_writer.add_scalar('Dev_US_reward', US_reward, epoch)
                self.tb_writer.add_scalar('Avg_turns', avg_turn, epoch)
                self.tb_writer.add_scalar('Match', match, epoch)
                self.tb_writer.add_scalar('Success', success, epoch)


    def run_RL_epoch(self):
        avg_US_reward=0
        avg_DS_reward=0
        for _ in range(cfg.rl_dial_per_epoch//cfg.interaction_batch_size):
            self.DS.eval()
            self.US.eval()
            gen_batch, US_reward_batch, DS_reward_batch, _ , _=\
                    self.interact_by_batch(cfg.interaction_batch_size, return_reward=True)
            avg_US_reward+=sum([np.mean(reward) for reward in US_reward_batch])
            avg_DS_reward+=sum([np.mean(reward) for reward in DS_reward_batch])
            self.DS.train()
            self.US.train()
            # Two different tokenizer
            gen_batch_ids=self.reader.convert_batch_tokens_to_ids(gen_batch, self.DS_tok)
            ds_turn_batches, ds_label_batches, ds_reward_batches = self.reader.transpose_ds_turn_batch(
                gen_batch_ids, DS_reward_batch)
            for turn_batch, label_batch, reward_batch in zip(ds_turn_batches, ds_label_batches, ds_reward_batches):
                if self.global_output>0:
                    pass
                    #logging.info(self.DS_tok.decode(list(turn_batch[0])))
                    #self.global_output-=1
                input_tensor = torch.from_numpy(turn_batch).long().to(self.DS.device)
                label_tensor = torch.from_numpy(label_batch).long().to(self.DS.device)
                outputs=self.DS(input_tensor)
                loss=self.calculate_loss(outputs, label_tensor, reward_batch)
                loss.backward()
                if cfg.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.DS.parameters(), 5.0)
                # we optimize after every minibatch
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                    lr=self.optimizer.param_groups[0]["lr"]
                    if lr>=2.8e-5 and cfg.ctrl_lr:
                        cfg.use_scheduler=False
                        self.scheduler=None
                self.optimizer.zero_grad()
                self.tb_writer.add_scalar('DS-lr', self.optimizer.param_groups[0]["lr"], self.DS_training_steps)
                self.DS_training_steps+=1
            gen_batch_ids=self.reader.convert_batch_tokens_to_ids(gen_batch, self.US_tok)
            us_turn_batches, us_label_batches, us_reward_batches = self.reader.transpose_us_turn_batch(
                gen_batch_ids, US_reward_batch, self.US_tok)
            for turn_batch, label_batch, reward_batch in zip(us_turn_batches, us_label_batches, us_reward_batches):
                if self.global_output>0:
                    pass
                    #logging.info(self.US_tok.decode(list(turn_batch[0])))
                    #self.global_output-=1
                input_tensor = torch.from_numpy(turn_batch).long().to(self.US.device)
                label_tensor = torch.from_numpy(label_batch).long().to(self.US.device)
                outputs=self.US(input_tensor)
                loss=self.calculate_loss(outputs, label_tensor, reward_batch)
                loss.backward()
                if cfg.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.US.parameters(), 5.0)
                self.optimizer_us.step()
                if self.scheduler_us:
                    self.scheduler_us.step()
                    lr=self.optimizer_us.param_groups[0]["lr"]
                    if lr>=2.8e-5 and cfg.ctrl_lr:
                        cfg.use_scheduler=False
                        self.scheduler_us=None
                self.optimizer_us.zero_grad()
                self.tb_writer.add_scalar('US-lr', self.optimizer_us.param_groups[0]["lr"], self.US_training_steps)
                self.US_training_steps+=1
        avg_DS_reward/=cfg.rl_dial_per_epoch
        avg_US_reward/=cfg.rl_dial_per_epoch
        #self.global_output=1
        return avg_DS_reward, avg_US_reward
    
    def evaluation(self):
        logging.info('DS path:{}, US path:{}'.format(cfg.DS_path, cfg.US_path))
        pointer=0
        dial_id_batches=[]
        while(pointer<=len(self.reader.test_list)):
            if pointer+cfg.interaction_batch_size<=len(self.reader.test_list):
                dial_id_batches.append(self.reader.test_list[pointer:pointer+cfg.interaction_batch_size])
            else:
                dial_id_batches.append(self.reader.test_list[pointer:])
            pointer+=cfg.interaction_batch_size
        return self.validate(dial_id_batches)


    def validate(self, dial_id_batches=None, init_goal_batches=None):
        logging.info("Start validation")
        avg_US_reward=0
        avg_DS_reward=0
        success=0
        match=0
        total=0
        avg_turn=0
        all_dials=[]
        st=time.time()
        if init_goal_batches is None:
            for dial_id_batch in dial_id_batches:
                total+=len(dial_id_batch)
                gen_batch, US_reward_batch, DS_reward_batch, batch_success, batch_match=\
                    self.interact_by_batch(len(dial_id_batch), dial_id_batch, return_reward=True)
                avg_turn+=sum([len(dial) for dial in gen_batch])
                avg_US_reward+=sum([np.mean(reward) for reward in US_reward_batch])
                avg_DS_reward+=sum([np.mean(reward) for reward in DS_reward_batch])
                success+=batch_success
                match+=batch_match
                all_dials+=gen_batch
        else:
            for init_goal_batch in init_goal_batches:
                total+=len(init_goal_batch)
                gen_batch, US_reward_batch, DS_reward_batch, batch_success, batch_match=\
                    self.interact_by_batch(len(init_goal_batch), init_goal_batch=init_goal_batch, return_reward=True)
                avg_turn+=sum([len(dial) for dial in gen_batch])
                avg_US_reward+=sum([np.mean(reward) for reward in US_reward_batch])
                avg_DS_reward+=sum([np.mean(reward) for reward in DS_reward_batch])
                success+=batch_success
                match+=batch_match
                all_dials+=gen_batch
        success/=total
        match/=total
        avg_US_reward/=total
        avg_DS_reward/=total
        avg_turn/=total
        #print('Avg_US_reward:{:3f}, avg_DS_reward:{:3f}, avg turns:{}, success rate:{:2f}, \
         #   match rate:{:.2f}'.format(avg_US_reward, avg_DS_reward, avg_turn, success, match))
        logging.info('Validation dialogs:{},  time:{:.2f} min'.format(total, (time.time()-st)/60))
        logging.info('Avg_US_reward:{:3f}, avg_DS_reward:{:3f}, avg turns:{}, success rate:{:2f}, \
            match rate:{:.2f}'.format(avg_US_reward, avg_DS_reward, avg_turn, success, match))
        if os.path.exists(cfg.exp_path):
            json.dump(all_dials, open(os.path.join(cfg.exp_path, 'validate_result.json'), 'w'), indent=2)
        return avg_DS_reward, avg_US_reward, avg_turn, success, match

    def save_model(self):
        self.DS.save_pretrained(os.path.join(cfg.exp_path,'best_DS'))
        self.DS_tok.save_pretrained(os.path.join(cfg.exp_path,'best_DS'))
        if cfg.joint_train:
            self.US.save_pretrained(os.path.join(cfg.exp_path,'best_US'))
            self.US_tok.save_pretrained(os.path.join(cfg.exp_path,'best_US'))

    def get_optimizers(self, num_dials, model):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)
        num_training_steps = num_dials*cfg.epoch_num // cfg.training_batch_size
        num_warmup_steps = cfg.warmup_steps if cfg.warmup_steps >= 0 else int(num_training_steps*cfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,\
            num_training_steps=num_training_steps) if cfg.use_scheduler else None
        return optimizer, scheduler

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

    def interact_by_batch(self, batch_size=cfg.interaction_batch_size, dial_id_batch=None, init_goal_batch=None, return_reward=False):
        max_turns=20
        gen_batch=[[] for _ in range(batch_size)]
        end_batch=[0 for _ in range(batch_size)]
        gpan_batch=[]
        goal_batch=[]# current goal batch
        bs_max_len=50
        act_max_len=20
        resp_max_len=60
        if dial_id_batch is None:
            dial_id_batch=random.sample(self.reader.train_list, batch_size)
        self.goal_list_batch=[[] for _ in range(batch_size)]
        if init_goal_batch is None:
            init_goal_batch=[]
            for batch_id, dial_id in enumerate(dial_id_batch):
                init_goal_batch.append(self.reader.data[dial_id]['goal'])
                goal=self.reader.goal_norm(self.reader.data[dial_id]['goal'])
                goal_batch.append(goal)
                self.goal_list_batch[batch_id].append(goal)
                gpan='<sos_g> '+self.reader.goal_to_gpan(goal)+' <eos_g>'
                gpan_batch.append(gpan)
        else:
            for batch_id, init_goal in enumerate(init_goal_batch):
                goal=self.reader.goal_norm(init_goal)
                goal_batch.append(goal)
                self.goal_list_batch[batch_id].append(goal)
                gpan='<sos_g> '+self.reader.goal_to_gpan(goal)+' <eos_g>'
                gpan_batch.append(gpan)
        self.turn_domain_batch=['' for _ in range(batch_size)]
        pv_resp_batch=None
        pv_bspn_batch=None
        for i in range(max_turns):

            if i>0: # update goals
                for batch_id, goal in enumerate(goal_batch):
                    pv_constraint=self.reader.bspan_to_constraint_dict(pv_bspn_batch[batch_id])
                    pv_user_act_dict=self.reader.aspan_to_act_dict(pv_user_act_batch[batch_id], side='user')
                    goal=self.reader.update_goal(goal,pv_user_act_dict,pv_constraint)
                    goal_batch[batch_id]=goal
                    self.goal_list_batch[batch_id].append(goal)
                    gpan_batch[batch_id]='<sos_g> '+self.reader.goal_to_gpan(goal)+' <eos_g>'
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
            pv_user_act_batch=user_act_batch

            # collect dialogs and judge stop
            for batch_id in range(batch_size):
                user_act=user_act_batch[batch_id]
                aspn=aspn_batch[batch_id]
                goal=goal_batch[batch_id]
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
                if (set(user_act.split()).issubset(self.end_tokens) and set(aspn.split()).issubset(self.end_tokens)) or goal=={}:
                    end_batch[batch_id]=1
            if all(end_batch):
                break
        if return_reward:
            US_reward_batch=[]
            DS_reward_batch=[]
            total_success=0
            total_match=0
            for init_goal, goal_list, gen_dial in zip(init_goal_batch, self.goal_list_batch, gen_batch):
                US_reward_batch.append(self.get_US_reward(gen_dial, goal_list))
                DS_reward_batch.append(self.get_DS_reward(gen_dial, init_goal))
                success, match=self.get_metrics(init_goal, gen_dial)
                total_success+=success
                total_match+=match
            
            return gen_batch, US_reward_batch, DS_reward_batch, total_success, total_match

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
                for u, b, db in zip(user_batch, bspn_batch, db_batch):
                    contexts.append(u + b + db + '<sos_a>')
            else:
                for u, b, db, a in zip(user_batch, bspn_batch, db_batch, aspn_batch):
                    contexts.append(u + b + db + a + '<sos_r>')
        else:
            if bspn_batch is None:
                for pv_b, pv_r, u in zip(pv_bspn_batch, pv_resp_batch, user_batch):
                    contexts.append(pv_b + pv_r + u + '<sos_b>')
            elif aspn_batch is None:
                for pv_b, pv_r, u, b, db in zip(pv_bspn_batch, pv_resp_batch, user_batch, bspn_batch, db_batch):
                    contexts.append(pv_b + pv_r + u + b + db + '<sos_a>')
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

    def get_US_reward(self, dial, goal_list):
        turn_num=len(dial)
        rewards=[]
        global_reward = 10*self.goal_complete_rate(goal_list[0], goal_list[-1])-turn_num
        pv_sys_act=None
        user_act_list=[]
        for turn, goal in zip(dial, goal_list):
            reqt_reward=0
            goal_reward=0
            repeat_reward=0
            end_reward=0
            user_act=self.reader.aspan_to_act_dict(turn['usr_act'], side='user')
            if cfg.add_end_reward:
                if user_act=={} and goal!={}: # user act is empty but goal is not, punish
                    for domain in goal:
                        for intent, sv in goal[domain].items():
                            if intent=='request':
                                for s in sv:
                                    if pv_sys_act is None or domain not in pv_sys_act or \
                                        intent not in pv_sys_act[domain] or s not in pv_sys_act[domain][intent]:
                                        end_reward-=1
                            else:
                                end_reward-=len(sv)
            if pv_sys_act:
                for domain in pv_sys_act:
                    if 'request' in pv_sys_act[domain]:
                        if domain not in user_act or 'inform' not in user_act[domain]:
                            reqt_reward-=len(pv_sys_act[domain]['request'])
                            continue
                        for slot in pv_sys_act[domain]['request']:
                            if slot in user_act[domain]['inform']:
                                reqt_reward+=1
                            else:
                                reqt_reward-=1
            for domain in user_act:
                for intent, sv in user_act[domain].items():
                    if domain not in goal or intent not in goal[domain]:
                        goal_reward-=len(sv)
                        continue
                    if isinstance(sv, list):
                        for slot in sv:
                            if slot in goal[domain][intent]:
                                goal_reward+=1
                            else:
                                goal_reward-=1
                    elif isinstance(sv, dict):
                        for slot, value in sv.items():
                            if slot not in goal[domain][intent]:
                                goal_reward-=1
                            elif value!=goal[domain][intent][slot]:
                                goal_reward-=1
                            else:
                                goal_reward+=1
            if user_act in user_act_list: # repeat the same action
                repeat_reward-=5
            user_act_list.append(user_act)
            pv_sys_act=self.reader.aspan_to_act_dict(turn['aspn'], side='sys')
            if cfg.non_neg_reward:
                rewards.append(max(min(reqt_reward + goal_reward + repeat_reward + end_reward + global_reward, 10),0))
            else:
                rewards.append(max(min(reqt_reward + goal_reward + repeat_reward + end_reward + global_reward, 10),-5))
        return rewards

    def get_DS_reward(self, dial, init_goal):
        turn_num=len(dial)
        rewards=[]
        sys_act_list=[]
        success, match=self.get_metrics(init_goal, dial)
        if success==1:
            global_reward=10-turn_num
        else:
            global_reward=7.5*match-turn_num
        for turn in dial:
            reqt_reward=0
            repeat_reward=0
            user_act=self.reader.aspan_to_act_dict(turn['usr_act'], side='user')
            sys_act=self.reader.aspan_to_act_dict(turn['aspn'], side='sys')
            for domain in user_act:
                if 'request' in user_act[domain]:
                    if domain not in sys_act or 'inform'  not in sys_act[domain]:
                        reqt_reward-=len(user_act[domain]['request'])
                        continue
                    for slot in user_act[domain]['request']:
                        if slot in sys_act[domain]['inform']:
                            reqt_reward+=1
                        else:
                            reqt_reward-=1
            if sys_act in sys_act_list:
                repeat_reward-=5
            sys_act_list.append(sys_act)
            if cfg.non_neg_reward:
                rewards.append(max(min(reqt_reward + repeat_reward + global_reward, 10),0))
            else:
                rewards.append(max(min(reqt_reward + repeat_reward + global_reward, 10),-5)) #[-5, 10]
        return rewards
        
    def goal_complete_rate(self, goal, final_goal):
        total_slot_num=0
        incomp_slot_num=0
        for domain in goal:
            for intent, sv in goal[domain].items():
                total_slot_num+=len(sv)
        if final_goal!={}:
            for domain in final_goal:
                for intent, sv in final_goal[domain].items():
                    incomp_slot_num+=len(sv)
        comp_rate=(total_slot_num-incomp_slot_num)/total_slot_num
        return comp_rate

    def get_metrics(self, init_goal, dial):
        reqs = {}
        goal = {}
        counts = {}
        for req in self.evaluator.requestables:
            counts[req+'_total'] = 0
            counts[req+'_offer'] = 0
        for domain in ontology.all_domains:
            if init_goal.get(domain):
                true_goal = init_goal
                goal = self.evaluator._parseGoal(goal, true_goal, domain)
        for domain in goal.keys():
            reqs[domain] = goal[domain]['requestable']
        success, match, _, _ = self.evaluator._evaluateGeneratedDialogue(dial, goal, reqs, counts)
        return success, match

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

    def calculate_loss(self, outputs, labels, rewards):
        # logits: B, T, V
        # labels: B, T
        # rewards: B
        batch_size=labels.size(0)
        lm_logits = outputs[0]

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=cfg.pad_id, reduction='sum')
        loss=0
        for i in range(batch_size):
            logit=shift_logits[i,:,:]
            label=shift_labels[i,:]
            reward=rewards[i]
            loss += reward*loss_fct(logit.view(-1, logit.size(-1)), label.view(-1))

        # avg loss
        not_ignore = shift_labels.ne(cfg.pad_id)
        num_targets = not_ignore.long().sum().item()

        loss /= num_targets
        return loss



def fix_seed():
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    parse_arg_cfg(args)
    cfg.exp_path=os.path.join(cfg.rl_save_path,cfg.exp_no)
    if not os.path.exists(cfg.exp_path):
        os.mkdir(cfg.exp_path)
    if 'test' in args.mode:
        cfg.eval_load_path=cfg.DS_path
    cfg._init_logging_handler(args.mode)
    fix_seed()
    session=turn_level_session(cfg.DS_path, cfg.US_path, cfg.DS_device, cfg.US_device)
    if 'train' in args.mode:
        session.run_RL()
    else:
        cfg.exp_path=cfg.DS_path
        session.evaluation()