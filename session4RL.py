from convlab2.util.analysis_tool.analyzer import Analyzer
from convlab2.nlu.jointBERT.multiwoz import BERTNLU
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.nlg.template.multiwoz import TemplateNLG
from numpy.core.fromnumeric import repeat
#from convlab2.e2e.damd.multiwoz import Damd
from model_LsGPT import *
from train_semi import Modal, parse_arg_cfg
from config import global_config as cfg
from rl_utils.checkInfoTurn import formKeySlot
from user_gpt import GPT2Agent
from eval import MultiWozEvaluator
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from convlab2.dialog_agent import PipelineAgent, BiSession
import ontology
import sys
import os
import copy
import random
import shutil
import math
import logging
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
slot_map={'Addr':'address', 'Post':'postcode','Ref':'reference','Depart':'departure','Dest':'destination'}
def set_seed(r_seed):
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

class RL_sess(Analyzer, Modal):
    def __init__(self,user_agent, sys_agent):
        if cfg.rl_with_us:
            Analyzer.__init__(self,user_agent)
            self.sess=self.build_sess(sys_agent)
            self.user_agent=user_agent
            self.us_model=self.user_agent.model
        #Modal.__init__(self)
        self.sys_agent=sys_agent
        self.tokenizer=sys_agent.tokenizer
        self.reader=sys_agent.reader
        self.evaluator = MultiWozEvaluator(self.reader)
        self.model=sys_agent.model
        self.key_slot=formKeySlot()
        self.get_special_id()
        self.obs=2#observes
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=cfg.pad_id, reduction='sum')
        json.dump(cfg.__dict__,open(os.path.join(cfg.exp_path,'cfg_all.json'),'w'),indent=2)
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
        cfg.origin_batch_size=cfg.batch_size # 4

    def run_RL(self):
        self.optimizer, self.scheduler=self.get_sep_optimizers(num_dials=cfg.rl_dial_per_epoch, model=self.model)
        if cfg.joint_train:
            self.optimizer_us, self.scheduler_us=self.get_sep_optimizers(num_dials=cfg.rl_dial_per_epoch, model=self.us_model)
        if cfg.init_eval:
            if cfg.validate_mode=='offline':
                dev_reward, metrics=self.validate()
                logging.info('Initial dev reward:{}, metrics:{}'.format(dev_reward, metrics))
            else:
                DS_reward, US_reward, metrics =self.validate()
                logging.info('Initial DS reward:{}, US_reward:{}, metrics:{}'.format(DS_reward, US_reward, metrics))
        
        max_score=0
        early_stop_count=cfg.early_stop_count
        weight_decay_count=cfg.weight_decay_count
        lr=cfg.lr
        cfg.batch_size=cfg.batch_size*cfg.gradient_accumulation_steps # 4*8
        self.all_batches = self.reader.get_batches('train')
        if cfg.traverse_data:
            cfg.rl_dial_per_epoch=len(self.reader.train)
        logging.info('Dialogs per rl epoch:{}'.format(cfg.rl_dial_per_epoch))
        for epoch in range(cfg.epoch_num):
            st=time.time()
            # shuffle train-set goal before every epoch
            if cfg.rl_with_us:
                random.shuffle(self.sess.user_agent.train_list)
                self.sess.user_agent.goal_pointer=0
            training_reward=self.run_RL_epoch()
            logging.info('Epoch:{}, time:{:.3f} min'.format(epoch, (time.time()-st)/60))
            if cfg.validate_mode=='offline':
                dev_reward, metrics=self.validate()
                (inform, success, bleu, score)=metrics
                # dev_reward:(r_bs, r_act, r_resp)
                eval_metric=metrics[3]
                logging.info('Dev reward:{}, metrics:{}'.format(dev_reward, metrics))
            else:
                DS_reward, US_reward, metrics =self.validate()
                (inform, success,score)=metrics
                eval_metric=(DS_reward+US_reward)/2
                logging.info('DS reward:{}, US_reward:{}, metrics:{}'.format(DS_reward, US_reward, metrics))

            logging.info('Average reward of epoch {}: {}'.format(epoch, training_reward))

            if eval_metric>max_score:
                max_score=eval_metric
                self.save_model()
                print('model saved in ', cfg.exp_path)
                early_stop_count=cfg.early_stop_count
            else:
                early_stop_count-=1
                weight_decay_count-=1
            if early_stop_count==0 and cfg.early_stop:
                print('early stop')
                break
            if weight_decay_count==0 and not cfg.use_scheduler:
                lr=lr*cfg.lr_decay
                for group in self.optimizer.param_groups:
                    group['lr'] = lr
                if cfg.joint_train:
                    for group in self.optimizer_us.param_groups:
                        group['lr'] = lr
                print("learning rate decay to {}".format(lr))
                weight_decay_count = cfg.weight_decay_count
                if lr<1e-9:
                    print('learning rate too small, break')
                    break

            if self.tb_writer:
                self.tb_writer.add_scalar('Avg_training_reward', training_reward, epoch)
                if cfg.validate_mode=='offline':
                    self.tb_writer.add_scalar('Reward1', dev_reward[0], epoch)
                    self.tb_writer.add_scalar('Reward2', dev_reward[1], epoch)
                    self.tb_writer.add_scalar('Reward3', dev_reward[2], epoch)
                    self.tb_writer.add_scalar('BLEU', bleu, epoch)
                else:
                    self.tb_writer.add_scalar('DS_reward', DS_reward, epoch)
                    self.tb_writer.add_scalar('US_reward', US_reward, epoch)
                self.tb_writer.add_scalar('Inform', inform, epoch)
                self.tb_writer.add_scalar('Success', success, epoch)
                self.tb_writer.add_scalar('Score', score, epoch)


        #print('Final offline evaluation on test')
        #_=self.validate(data='test')
        #print('Final online evaluation on test')
        #_=self.validate_online(set_name='test')

    def save_model(self):
        self.model.save_pretrained(os.path.join(cfg.exp_path,'best_DS'))
        self.tokenizer.save_pretrained(os.path.join(cfg.exp_path,'best_DS'))
        if cfg.joint_train:
            self.us_model.save_pretrained(os.path.join(cfg.exp_path,'best_US'))
            self.user_agent.tokenizer.save_pretrained(os.path.join(cfg.exp_path,'best_US'))

    def run_RL_epoch(self):
        update_count=0
        batch_count=0
        total_batch=len(self.all_batches)
        avg_sys_reward=0
        while(True):
            if update_count>=cfg.rl_dial_per_epoch:
                break
            self.model.eval()
            #rl training
            for i in range(cfg.rl_iterate_num):
                if cfg.rl_with_us:
                    #rl training with user simulator
                    dial_batch_large=self.generate_dials_with_us(dial_num=cfg.batch_size)
                else:
                    if cfg.traverse_data:
                        batch=self.all_batches[batch_count%total_batch]
                        batch_count+=1
                    else:
                        batch=random.sample(self.all_batches,1)[0]
                        #batch=copy.deepcopy(random.sample(self.all_batches,1)[0])
                    dial_batch_large=self.generate_dials_by_sampling(batch)#default generate one batch
                self.model.train()
                dial_batch=[]
                for i, dial in enumerate(dial_batch_large):
                    dial_batch.append(dial)
                    if len(dial_batch)==cfg.origin_batch_size or i==len(dial_batch_large)-1:
                        rl_loss, sys_reward=self.get_rl_loss(dial_batch, side='sys')
                        rl_loss.backward()
                        avg_sys_reward += sys_reward
                        if cfg.joint_train:
                            us_rl_loss, user_reward=self.get_rl_loss(dial_batch, side='usr')
                            us_rl_loss.backward()
                        update_count+=len(dial_batch)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                        dial_batch=[]
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                if cfg.joint_train:
                    self.optimizer_us.step()
                    if self.scheduler_us:
                        self.scheduler_us.step()
                    self.optimizer_us.zero_grad()
                update_count+=len(dial_batch_large)
                    
            '''
            #supervised training
            if cfg.rl_iterate:
                sup_batch=random.sample(self.all_batches,1)[0]
                loss=self.get_sup_loss(sup_batch)
                loss.backward()
                back_step+=1
                update_count+=len(sup_batch)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            
            if back_step%cfg.gradient_accumulation_steps==0 or update_count>cfg.rl_dial_per_epoch:
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                if cfg.joint_train:
                    self.optimizer_us.step()
                    if self.scheduler_us:
                        self.scheduler_us.step()
                    self.optimizer_us.zero_grad()
            '''
        avg_sys_reward/=update_count
        return avg_sys_reward
        


    def get_special_id(self):
        self.sos_b_id=self.tokenizer.convert_tokens_to_ids('<sos_b>')
        self.sos_a_id=self.tokenizer.convert_tokens_to_ids('<sos_a>')
        self.sos_r_id=self.tokenizer.convert_tokens_to_ids('<sos_r>')
        self.eos_b_id=self.tokenizer.convert_tokens_to_ids('<eos_b>')
        self.eos_a_id=self.tokenizer.convert_tokens_to_ids('<eos_a>')
        self.eos_r_id=self.tokenizer.convert_tokens_to_ids('<eos_r>')
        self.sos_db_id=self.tokenizer.convert_tokens_to_ids('<sos_db>')
        self.eos_db_id=self.tokenizer.convert_tokens_to_ids('<eos_db>')

    def get_sep_optimizers(self,num_dials,model):
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
        num_training_steps = num_dials*cfg.epoch_num // (cfg.gradient_accumulation_steps*cfg.origin_batch_size)
        num_warmup_steps = cfg.warmup_steps if cfg.warmup_steps >= 0 else int(num_training_steps*cfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,\
            num_training_steps=num_training_steps) if cfg.use_scheduler else None
        return optimizer, scheduler

    def validate(self, **kwargs):
        if cfg.validate_mode=='offline':
            return self.validate_offline()
        else:
            # evaluation by interaction
            # set your own dialog nums
            return self.validate_online(dial_num=100)
            

    def validate_online(self, set_name='dev', dial_num=None):# 'dev' or 'test'
        all_dials=[]
        dials_for_eval=[]
        st=time.time()
        set_file=self.reader.dev_list if set_name=='dev' else self.reader.test_list
        if dial_num:
            set_file=set_file[:dial_num]
        DS_reward=0
        US_reward=0
        result_path=os.path.join(cfg.eval_load_path, 'online_result.json')
        if os.path.exists(result_path) and 'test' in cfg.mode:
            logging.info('Using existing result file to evaluate')
            data=json.load(open(result_path,'r', encoding='utf-8'))
            for dial in data:
                DS_reward += np.mean(self.get_DS_reward(dial['log']))
                US_reward += np.mean(self.get_US_reward(dial['log']))
                for turn in dial['log']:
                    dials_for_eval.append(turn)
            logging.info('Online average reward of DS:{:.3f}, US:{:.3f}'.format(DS_reward/len(set_file),US_reward/len(set_file)))
            success, match, req_offer_counts, dial_num = self.evaluator.context_to_response_eval(dials_for_eval)           
            score=0.5*(success + match)
            logging.info('Online test on {}, time consuming:{:.2f} min\nsuccess:{:.2f}, inform:{:.2f}, '.format(set_name, (time.time()-st)/60, success, match))
            return (success, match, score)
        with torch.no_grad():
            for dial_id in set_file:
                sys_response = '' if self.user_agent.nlu else []
                self.sess.init_session(goal=self.reader.data[dial_id]['goal'])
                for i in range(30):# max turns:30
                    sys_response, user_response, session_over, reward = self.sess.next_turn(sys_response)
                    if session_over:
                        break
                history=self.sess.sys_agent.info_history
                for turn_id, turn in enumerate(history):
                    turn['dial_id']=dial_id
                    turn['resp_gen']=turn['resp']
                    turn['bspn_gen']=turn['bspn']
                    dials_for_eval.append(turn)
                
                for turn_id, _ in enumerate(history):
                    history[turn_id]['usr_intent']=self.sess.user_agent.action_history[turn_id]
                    history[turn_id]['usr_act']=self.sess.user_agent.aspn_history[turn_id]
                generated_dial={'goal':self.sess.user_agent.goal,'log':history}
                all_dials.append(generated_dial)
                DS_reward += np.mean(self.get_DS_reward(generated_dial['log']))
                US_reward += np.mean(self.get_US_reward(generated_dial['log']))

        
        logging.info('Online average reward of DS:{:.2f}, US:{:.2f}'.format(DS_reward/len(set_file),US_reward/len(set_file)))
        success, match, req_offer_counts, dial_num = self.evaluator.context_to_response_eval(dials_for_eval)           
        score=0.5*(success + match)
        logging.info('Online test on {}, time consuming:{:.2f} min\nsuccess:{:.2f}, inform:{:.2f}, '.format(set_name, (time.time()-st)/60, success, match))
        json.dump(all_dials, open(result_path, 'w'), indent=2)
        return DS_reward, US_reward, (success, match, score)

    def validate_offline(self, set_name='dev'):
        origin_batch_size=cfg.batch_size
        cfg.batch_size=cfg.eval_batch_size
        eval_batches=self.reader.get_batches(set_name)
        result_collection={}
        R1=0
        R2=0
        R3=0
        st=time.time()
        for batch in eval_batches:
            batch, batch_ids=self.generate_dials_by_sampling(batch, return_ids=True)          
            for dialog in batch:
                # if turn-level, reward:[(rb, ra, rr), (rb, ra, rr), ...]
                # else, reward:[r1, r2, r3, ...]
                reward=self.get_reward_of_sampling(dialog)
                if isinstance(reward, list):
                    if isinstance(reward[0], tuple):
                        reward=(sum([item[0] for item in reward]),
                        sum([item[1] for item in reward]),
                        sum([item[2] for item in reward])
                        )
                    else:
                        reward=sum(reward)
                if isinstance(reward, tuple):
                    R1+=reward[0]
                    R2+=reward[1]
                    R3+=reward[2]
                else:
                    R1+=reward
            #batch=self.reader.convert_batch_tokens_to_ids(batch)
            for dialog in batch_ids:
                result_collection.update(self.reader.inverse_transpose_turn(dialog))
        logging.info('Inference time:{:2f}'.format((time.time()-st)/60))
        results, field = self.reader.wrap_result_lm(result_collection)
        bleu, success, match = self.evaluator.validation_metric(results)
        score = 0.5 * (success + match) + bleu
        cfg.batch_size=origin_batch_size
        return (R1,R2, R3), (match, success, bleu, score)
                
    def generate_dials_with_us(self, dial_num=2):

        goal_seeds = [random.randint(1,100000) for _ in range(dial_num+30)]
        all_dials=[]
        all_rewards=[]
        with torch.no_grad():
            while(True):
                sys_response = '' if self.user_agent.nlu else []
                self.sess.init_session()
                rewards=[]
                for i in range(40):# max turns:40
                    sys_response, user_response, session_over, reward = self.sess.next_turn(sys_response)
                    rewards.append(reward)
                    if session_over:
                        break
                history=self.sess.sys_agent.info_history
                for turn_id, _ in enumerate(history):
                    history[turn_id]['usr_intent']=self.sess.user_agent.action_history[turn_id]
                    history[turn_id]['usr_act']=self.sess.user_agent.aspn_history[turn_id]
                turn_num=len(history)
                if turn_num<20:
                    all_dials.append({'goal':self.sess.user_agent.goal,'log':history})
                if len(all_dials)==dial_num:
                    break
        return all_dials
    
    def generate_dials_by_sampling(self, batch, return_ids=False):
        #use the function generate_batch in train_semi.py
        self.device1=self.model.device
        self.PrioriModel=self.model
        batches=[batch]
        new_batch=[]
        for batch in batches:
            try:
                if cfg.turn_level:
                    batch=self.generate_batch_turn_level(batch)
                else:
                    batch=self.generate_batch(batch)
                new_batch+=batch
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    logging.info("WARNING: ran out of memory during generation and batch will be divided by half, batch size:{}, turn num:{}"\
                        .format(len(batch),len(batch[0])))
                    if hasattr(torch.cuda, 'empty_cache'):
                        with torch.cuda.device(self.model.device):
                            torch.cuda.empty_cache()
                    batches+= [ batch[:len(batch)//2], batch[len(batch)//2:] ]
                else:
                    logging.info(str(exception))
                    raise exception
        batch=new_batch
        batch_ids=copy.deepcopy(batch)
        for i, dial in enumerate(batch):
            for j,turn in enumerate(dial):
                for key in turn:
                    if key in ['user','bspn','bspn_gen','aspn','aspn_gen','resp','resp_gen','db','db_gen']:
                        batch[i][j][key]=self.tokenizer.decode(batch[i][j][key])
        if return_ids:
            return batch, batch_ids
        else:
            return batch         
        
        
    
    def get_rl_loss(self, gen_dial_batch, side='sys'):
        # side: 'sys'/'usr'
        intent_key=side+'_intent'
        rl_loss = 0
        n_token = 0
        rewards=[]
        turn_nums=[]
        avg_reward=0
                
        for gen_dial in gen_dial_batch:
            if cfg.rl_with_us:
                if side=='sys':
                    reward=self.get_DS_reward(gen_dial['log'])
                elif side=='usr':
                    reward=self.get_US_reward(gen_dial['log'])
            else:
                reward=self.get_reward_of_sampling(gen_dial)
            rewards.append(reward)
            turn_nums.append(len(gen_dial))
        if self.obs>0:
            logging.info('Example rewards of one batch :{}'.format(rewards))
            self.obs-=1
        
        dial_batch=self.reader.convert_batch_tokens_to_ids(gen_dial_batch)

        if cfg.turn_level:
            if side=='sys':
                dial_batch=self.reader.transpose_batch(dial_batch)
                pv_batch = None
                for turn_num, turn_batch in enumerate(dial_batch):
                    first_turn = (turn_num == 0)
                    inputs, _, rl_labels = self.reader.convert_batch_turn(turn_batch, pv_batch, first_turn,rl_train=True, mode='gen')
                    pv_batch = self.reader.get_pv_batch(pv_batch, turn_batch['user'],
                        turn_batch['resp_gen'], turn_batch['bspn_gen'])
                    inputs=torch.from_numpy(inputs['contexts_np']).long()
                    #labels=torch.from_numpy(labels['contexts_np']).long()
                    rl_labels=torch.from_numpy(rl_labels['contexts_np']).long()
                    outputs=self.model(inputs.to(self.model.device))
                    rl_loss=self.calculate_loss_by_turn(inputs, outputs[0], rl_labels, rewards, turn_num)
            for reward in rewards:
                for turn_reward in reward:
                    avg_reward += sum(turn_reward)
                    
            
        else:
            if side=='sys':
                seq_batch, label_batch=self.reader.convert_batch_session(dial_batch, rl_train=True)
            elif side=='usr':
                seq_batch, label_batch=self.reader.convert_us_batch_session(dial_batch)
            
            seq_batch_tensor=torch.from_numpy(seq_batch['contexts_np']).long()
            label_batch_tensor=torch.from_numpy(label_batch['contexts_np']).long()
            outputs = self.model(seq_batch_tensor.to(self.model.device))
            logits=outputs[0]#B,T,V
            rl_loss=self.calculate_loss_by_session(logits, label_batch_tensor, rewards)
            
            for reward in rewards:
                if isinstance(reward, list):
                    avg_reward += np.mean(reward)
                else:
                    avg_reward += reward
            #avg_reward=avg_reward/len(rewards)
        return rl_loss, avg_reward

    def get_sup_loss(self, dial_batch):
        inputs, labels = self.reader.convert_batch_session(dial_batch)
        inputs_tensor=torch.from_numpy(inputs['contexts_np']).long()
        labels_tensor=torch.from_numpy(labels['contexts_np']).long()
        outputs=self.model(inputs_tensor.to(self.model.device))
        loss=self.calculate_loss_and_accuracy(outputs,labels_tensor.to(self.model.device))
        return loss


    def calculate_loss_by_session(self, logits, labels, rewards):
        # session-level
        # rewards: [r1, r2, r3, ..., rT]
        batch_size=logits.size(0)
        rl_loss=0
        n_token=0
        for batch_idx in range(batch_size):
            label=labels[batch_idx]#T
            reward=rewards[batch_idx]
            if cfg.turn_level_reward:
                label_rev=label.tolist()[::-1]#Traverse in reverse order, because the previous turns may have been pre-truncated
                label_rev=torch.tensor(label_rev).ne(cfg.pad_id).long().tolist()
                loc1=0
                loc2=0
                count=0
                while(1):
                    if 1 not in label_rev:
                        break
                    loc1=label_rev.index(1)+loc2
                    label_rev=label_rev[loc1-loc2:]
                    if 0 not in label_rev:
                        break
                    loc2=label_rev.index(0)+loc1
                    label_rev=label_rev[loc2-loc1:]
                    if loc2-loc1<4:
                        break
                    logit=logits[batch_idx,-(loc2+1):-(loc1+2),:]#T,V
                    label_for_logit=label[-loc2:-(loc1+1)]#T
                    loss = self.loss_fct(logit.view(-1,logit.size(-1)), label_for_logit.view(-1).to(self.model.device))
                    n_token+=loc2-loc1
                    count+=1
                    rl_loss+=reward[-count]*loss
            else:
                logit=logits[batch_idx,:-1,:].contiguous()
                label=label[1:].contiguous()
                rl_loss+=reward*self.loss_fct(logit.view(-1,logit.size(-1)),label.view(-1).to(self.model.device))
                not_ignore = label.ne(cfg.pad_id)
                n_token += not_ignore.long().sum().item()
        rl_loss = rl_loss/n_token
        return rl_loss

    def calculate_loss_by_turn(self, inputs, logits, labels, rewards, turn_num):
        # turn-level
        # inputs: B, T
        # logits: B, T, V
        # labels: B, T
        # rewards: [[(r_b, r_a, r_r),(r_b, r_a, r_r),...],[],...,[]]
        rl_loss=0
        batch_size=logits.size(0)
        n_tokens=0
        for batch_idx in range(batch_size):
            label=list(labels[batch_idx, :])
            rev_label=label[::-1]
            # find bspn
            for label_id in [1,2,3]:
                # find bspn, aspn and resp
                start_idx=label.index(label_id) # <sos_b>
                end_idx=rev_label.index(label_id) # <eos_b>
                if end_idx==0:
                    bspn_logit=logits[batch_idx, start_idx:-1, :]
                    bspn_target=inputs[batch_idx, start_idx+1:]
                else:
                    bspn_logit=logits[batch_idx, start_idx:-(end_idx+1), :]
                    bspn_target=inputs[batch_idx, start_idx+1:-end_idx]
                loss = self.loss_fct(bspn_logit.view(-1,bspn_logit.size(-1)), bspn_target.view(-1).to(self.model.device))
                n_tokens+=len(bspn_target)
                reward=rewards[batch_idx][turn_num][label_id-1]
                rl_loss+= reward*loss
        rl_loss=rl_loss/n_tokens
        return rl_loss

    def get_reward(self,gen_dial):
        # turn level reward: [r1,r2,r3...]
        # session level reward: r
        if cfg.turn_level_reward:

            R1=self.get_entity_provide_reward(gen_dial)
            R2=self.get_miss_answer_reward(gen_dial)
            R3=self.get_repeat_ask_reward(gen_dial)
            sys_reward=[r1+r2+r3 for r1,r2,r3 in zip(R1,R2,R3)]
            #gen_dial['sys_reward']=sys_reward
            #fix reward: nonnegative
            for i, reward in enumerate(sys_reward):
                if reward<0:
                    sys_reward[i]=0
                elif reward>=0:
                    sys_reward[i]=1
        else:
            if self.sess.evaluator.task_success():
                sys_reward=20
            elif self.sess.user_agent.policy.policy.goal.task_complete():
                sys_reward=10
            else:
                sys_reward=0
            if sys_reward>0:
                sys_reward=max(0, sys_reward-len(gen_dial))/5
        return sys_reward
    
    def get_reward_of_sampling(self, gen_dial):
        
        def request_score(gen, truth):
            tp, fp, fn = 0, 0, 0
            truth_req, gen_req = set(), set()
            for w in gen.split():
                if '[value_' in w and 'name]' not in w:
                    gen_req.add(w[w.index('[')+1:w.index(']')].split('_')[1])
            for w in truth.split():
                if '[value_' in w and 'name]' not in w:
                    truth_req.add(w[w.index('[')+1:w.index(']')].split('_')[1])
            for req in gen_req:
                if req in truth_req:
                    tp += 1
                else:
                    fp += 1
            for req in truth_req:
                if req not in gen_req:
                    fn += 1
            #print(gen_req, truth_req)
            precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
            #print('precision:', precision, 'recall:', recall)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            return f1
        
        if not cfg.turn_level_reward:#combined score as reward 
            reqs = {}
            goal = {}
            counts = {}
            for req in self.evaluator.requestables:
                counts[req+'_total'] = 0
                counts[req+'_offer'] = 0
            dial_id=gen_dial[0]['dial_id']
            if '.json' not in dial_id and '.json' in list(self.evaluator.all_data.keys())[0]:
                dial_id = dial_id + '.json'
            for domain in ontology.all_domains:
                if self.evaluator.all_data[dial_id]['goal'].get(domain):
                    true_goal = self.evaluator.all_data[dial_id]['goal']
                    goal = self.evaluator._parseGoal(goal, true_goal, domain)
            # print(goal)
            for domain in goal.keys():
                reqs[domain] = goal[domain]['requestable']
            success, match, _, _ = \
                self.evaluator._evaluateGeneratedDialogue(gen_dial, goal, reqs, counts, soft_acc=True, same_eval_as_cambridge=False)

            wrap_generated=[[turn['resp_gen']] for turn in gen_dial]
            wrap_truth=[[turn['resp']] for turn in gen_dial]
            bleu = self.evaluator.bleu_scorer.score(zip(wrap_generated, wrap_truth))
            #success is more important than match
            combined_score=bleu/100 + 0.5*success+0.5*match
            reward=1.5/(1+math.exp(-10*(combined_score-0.8)))#
            '''
            if self.obs>0:
                print('Inform:{:.2f}, Success:{:.2f}, BLEU:{:.2f}, Combined:{:.2f}, Reward:{:.2f}'\
                      .format(match, success, bleu, combined_score, reward))
                self.obs-=1
            '''
        else:
            bs_reward=self.get_dialog_state_reward(gen_dial) # bs_reward: [r1, r2, r3,...]
            act_reward=self.get_sys_act_reward(gen_dial)
            resp_reward=self.get_resp_reward(gen_dial)
            R=self.get_session_reward(gen_dial)# 0 ~ 1
            if self.obs>0:
                logging.info('Reward example. BS reward:{}, Act reward:{}, Resp reward:{}, Session reward:{}'.format(
                    bs_reward, act_reward, resp_reward, R))
            reward=[(r1+R, r2+R, r3+R) for r1, r2, r3 in zip(bs_reward, act_reward, resp_reward)]

        return reward

    def get_DS_reward(self, dial):
        rewards=[]
        for turn in dial:
            reqt_num=0
            info_num=0
            for usr_intent in turn['usr_intent']:
                if usr_intent[0]=='Request':
                    reqt_num+=1
                    domain=usr_intent[1]
                    slot=usr_intent[2]
                    for sys_intent in turn['sys_intent']:
                        # both inform and recommend can answer the inform
                        if sys_intent[:3]==['Inform', domain, slot] or sys_intent[:3]==['Recommend', domain, slot]:
                            info_num+=1
                            break
            if reqt_num==0:
                rewards.append(1)
            else:
                rewards.append((info_num/reqt_num-cfg.rate_th)/cfg.rate_th)
        '''
        repeat_info_rewards=self.get_repeat_info_reward(dial, side='sys')
        repeat_reqt_rewards=self.get_repeat_reqt_reward(dial, side='sys')
        if cfg.non_neg_reward:
            final_rewards=[max(r1+r2+r3,0) for r1, r2, r3 in zip(rewards, repeat_info_rewards, repeat_reqt_rewards)]
        else:
            final_rewards=[r1+r2+r3 for r1, r2, r3 in zip(rewards, repeat_info_rewards, repeat_reqt_rewards)]
        return final_rewards
        '''
        return rewards
    
    def get_US_reward(self,dial):
        rewards=[]
        pre_sys_intent=[]
        for turn in dial:
            reqt_num=0
            info_num=0
            if pre_sys_intent!=[]:
                for sys_intent in pre_sys_intent:
                    if sys_intent[0]=='Request':
                        reqt_num+=1
                        domain=sys_intent[1]
                        slot=sys_intent[2]
                        for usr_intent in turn['usr_intent']:
                            if usr_intent[:3]==['Inform', domain, slot]:
                                info_num+=1
                                break
            if reqt_num==0:
                rewards.append(1)
            else:
                rewards.append(2*info_num/reqt_num)

        repeat_info_rewards=self.get_repeat_info_reward(dial, side='usr')
        repeat_reqt_rewards=self.get_repeat_reqt_reward(dial, side='usr')
        if cfg.non_neg_reward:
            final_rewards=[max(r1+r2+r3,0) for r1, r2, r3 in zip(rewards, repeat_info_rewards, repeat_reqt_rewards)]
        else:
            final_rewards=[r1+r2+r3 for r1, r2, r3 in zip(rewards, repeat_info_rewards, repeat_reqt_rewards)]
        return final_rewards


    def get_repeat_info_reward(self, dial, side='usr'):
        rewards=[]
        side_key=side+'_intent'# usr_intent or sys_intent
        for turn_idx, turn in enumerate(dial):
            if turn_idx==0:
                rewards.append(0)
            if turn_idx>0:
                info_num=0
                repeat_info_num=0
                prev_turn=dial[turn_idx-1]
                for pre_intent in prev_turn[side_key]:
                    if pre_intent[0]=='Inform':
                        info_num+=1
                        for intent in turn[side_key]:
                            if intent==pre_intent:
                                repeat_info_num+=1
                                break
                if info_num==0:
                    rewards.append(0)
                else:
                    rewards.append(-repeat_info_num/info_num)
        return rewards

    def get_repeat_reqt_reward(self, dial, side='usr'):
        rewards=[]
        side_key=side+'_intent'# usr_intent or sys_intent
        for turn_idx, turn in enumerate(dial):
            if turn_idx==0:
                rewards.append(0)
            if turn_idx>0:
                reqt_num=0
                repeat_reqt_num=0
                prev_turn=dial[turn_idx-1]
                for pre_intent in prev_turn[side_key]:
                    if pre_intent[0]=='Request':
                        reqt_num+=1
                        for intent in turn[side_key]:
                            if intent[:3]==pre_intent[:3]:
                                repeat_reqt_num+=1
                                break
                if reqt_num==0:
                    rewards.append(0)
                else:
                    rewards.append(-repeat_reqt_num/reqt_num)
        return rewards


    def get_sys_act_reward(self, dial):
        sys_act_reward=[]
        for turn in dial:
            total=0
            tp=0
            act=self.reader.aspan_to_act_dict(turn['aspn'])
            act_gen=self.reader.aspan_to_act_dict(turn['aspn_gen'])
            for domain in act_gen:
                for intent in act_gen[domain]:
                    for slot in act_gen[domain][intent]:
                        total+=1
                        if domain in act:
                            if intent in act[domain]:
                                if slot in act[domain][intent]:
                                    tp+=1
            if total==0:
                reward=1
            else:
                reward=(tp/total-cfg.rate_th)/cfg.rate_th

            sys_act_reward.append(reward)
        return sys_act_reward
    
    def get_session_reward(self, gen_dial):
        reqs = {}
        goal = {}
        counts = {}
        for req in self.evaluator.requestables:
            counts[req+'_total'] = 0
            counts[req+'_offer'] = 0
        dial_id=gen_dial[0]['dial_id']
        if '.json' not in dial_id and '.json' in list(self.evaluator.all_data.keys())[0]:
            dial_id = dial_id + '.json'
        for domain in ontology.all_domains:
            if self.evaluator.all_data[dial_id]['goal'].get(domain):
                true_goal = self.evaluator.all_data[dial_id]['goal']
                goal = self.evaluator._parseGoal(goal, true_goal, domain)
        # print(goal)
        for domain in goal.keys():
            reqs[domain] = goal[domain]['requestable']
        success, match, _, _ = \
            self.evaluator._evaluateGeneratedDialogue(gen_dial, goal, reqs, counts, soft_acc=True, same_eval_as_cambridge=False)
        return success

    def get_dialog_state_reward(self, dial):
        dialog_state_reward=[]
        for turn in dial:
            bs=self.reader.bspan_to_constraint_dict(turn['bspn'])
            bs_gen=self.reader.bspan_to_constraint_dict(turn['bspn_gen'])
            total=0
            tp=0
            fp=0
            fn=0
            for domain in bs_gen:
                for slot in bs_gen[domain]:
                    total+=1
                    if domain in bs:
                        if slot in bs[domain]:
                            if bs_gen[domain][slot]==bs[domain][slot]:
                                tp+=1
                            else:
                                fp+=1
                        else:
                            fp+=1
                    else:
                        fp+=1
            for domain in bs:
                for slot in bs[domain]:
                    if domain in bs_gen:
                        if slot not in bs_gen[domain]:
                            fn+=1
                    else:
                        fn+=1
            if total==0:
                reward=1
            else:
                reward=(tp/total-cfg.rate_th)/cfg.rate_th
            dialog_state_reward.append(reward)
        return dialog_state_reward
    
    def get_resp_reward(self, dial):
        resp_reward=[]
        for turn in dial:
            resp_gen=turn['resp_gen']
            act=self.reader.aspan_to_act_dict(turn['aspn'])
            reward=0
            for domain in act:
                for intent in act[domain]:
                    if intent=='inform':
                        if 'name' in act[domain][intent] and '[value_name]' not in resp_gen:
                            reward+=cfg.resp_punish
                        if 'id' in act[domain][intent] and '[value_id]' not in resp_gen:
                            reward+=cfg.resp_punish
            reward=1 if reward==0 else reward
            resp_reward.append(reward)
        return resp_reward

    def get_entity_provide_reward(self, gen_dial, print_log=True):
        '''
        Description:
            get the turn level reward within a dialogue, (+) if entity is provided and (-) if entity is no provided
            this reward is domain-independent, and only affects the turns at info stage (which means book/reqt turns are not affected) 
        Return:
            a list (len=dial_len) of real number
        '''
        # trace turn domain within dialogue
        domain_prev = 'none'
        domain_history = []
        for turn in gen_dial:
            domain_history.append(turn['turn_domain'][0])
        # decide valid turns that influence entity provide
        domain_entityProvided = set(['taxi', 'police', 'hospital', 'general'])
        weight = []
        w = 1.
        #for act_usr, act_sys, turn_domain in zip(gen_dial['act_usr'], gen_dial['act_sys'], domain_history):
        for turn in gen_dial:
            turn_domain=turn['turn_domain'][0]
            if turn_domain != domain_prev: # domain transit
                w = 1.
            if turn_domain in ['taxi', 'police', 'hospital', 'general']: # no need reward for those domain
                weight.append(0)
            else:
                weight.append(w)

            # TODO: re-think this for domain transfer case
            if '[value_name]' in turn['resp'] or '[value_id]' in turn['resp']:
                domain_entityProvided.add(turn_domain)
                w = 0
            domain_prev = turn_domain

        # deal with domain without name provide since boundary between info and book/reqt is not decided yet
        #for side_idx, (act_usr, act_sys, turn_domain) in enumerate(zip(gen_dial['act_usr'], gen_dial['act_sys'], domain_history)):
        for turn_id, turn in enumerate(gen_dial):
            turn_domain=turn['turn_domain'][0]
            if turn_domain in domain_entityProvided: # done already
                continue
            '''
            decide turn stage within a domain, either info, book, reqt or none (for general domain)
            check by some rules:
                book: if usr informs any booking slot or if sys reqt any booking slot
                reqt: if usr reqt any reqt slot or if sys inform any reqt slot
            '''
            usr_intent=turn['usr_intent']
            sys_intent=turn['sys_intent']
            for intent in usr_intent:
                if intent[1]=='Booking' and intent[0]=='Inform':
                    weight[turn_id]=0
                elif intent[0]=='Request':
                    domain=intent[1].lower()
                    if domain in self.key_slot:
                        if 'reqt' in self.key_slot[domain]:
                            slot=slot_map.get(intent[2],intent[2].lower())
                            if slot in self.key_slot[domain]['reqt']:
                                weight[turn_id]=0
            for intent in sys_intent:
                if intent[1]=='Booking' and intent[0]=='Request':
                    weight[turn_id]=0
                elif intent[0]=='Inform':
                    domain=intent[1].lower()
                    if domain in self.key_slot:
                        if 'reqt' in self.key_slot[domain]:
                            slot=slot_map.get(intent[2],intent[2].lower())
                            if slot in self.key_slot[domain]['reqt']:
                                weight[turn_id]=0

        reward = []
        for side_idx, (w, turn_domain) in enumerate(zip(weight, domain_history)):
            if turn_domain in domain_entityProvided:
                r = cfg.entity_provide_reward
            else:
                r = cfg.no_entity_provide_reward
            reward.append(w*r)

        # trace reward
        '''
        if print_log:
            print('sys provide - dial name:', gen_dial['dial_name'])
            print('name provied:', domain_entityProvided-set(['taxi', 'police', 'hospital', 'general']))
            for side_idx, (w, r, usr_act, sys_act, turn_domain, usr_word, sys_word) in enumerate(zip(weight, reward, gen_dial['act_usr'], gen_dial['act_sys'], domain_history, gen_dial['word_usr'], gen_dial['word_sys'])):
                print('{}, w: {}, sys r: {}, d: {} | {} -> {}'.format(side_idx, w, r, turn_domain, usr_act, sys_act))
        '''
        return reward

    def get_repeat_ask_reward(self, gen_dial, print_log=True):
        '''
        Description:
            check if sys requests the informed slots
        Return:
            a list of real value rewards
        '''
        reward = []
        for turn in gen_dial:
        #for side_idx, (full_bs, act_usr, act_sys) in enumerate(zip(gen_dial['bs'], gen_dial['act_usr'], gen_dial['act_sys'])):
            r = 0 # assume no repeat ask until fine one
            cons=self.reader.bspan_to_constraint_dict(turn['bspn'])#{'hotel': {'stay': '3'}, 'restaurant': {'people': '4'}}
            flag=0
            for intent in turn['sys_intent']:
                if intent[0]=='Request':
                    flag=1
                    domain=intent[1].lower()
                    slot=slot_map.get(intent[2],intent[2].lower())
                    if domain=='train' and slot=='ticket':
                        slot='price'
                    if domain=='attraction' and slot=='fee':
                        slot='price'
                    if domain in ['booking','general']:
                        continue
                    if domain in cons:
                        if slot in cons[domain] and cons[domain][slot]!='':
                            r += cfg.repeat_ask_reward

            if flag and r == 0: # request right slots
                r = cfg.no_repeat_ask_reward
            reward.append(r)

        return reward

    def get_miss_answer_reward(self, gen_dial, print_log=True):
        reward = []
        for turn in gen_dial:
            r=0
            flag=0
            for intent in turn['usr_intent']:
                if intent[0]=='Request':
                    flag=1
                    slot=intent[2]
                    miss=True
                    for sys_intent in turn['sys_intent']:
                        if slot==sys_intent[2]:
                            miss=False
                            break
                    if miss:
                        r += cfg.miss_answer_reward
            if flag and r==0:
                r=cfg.no_miss_answer_reward

            reward.append(r)
        return reward

    def validate_us(self, data='dev'):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    parse_arg_cfg(args)
    if 'test' in cfg.mode:
        cfg.exp_path=cfg.DS_path
        print('DS path:', cfg.DS_path)
    else:
        cfg.exp_path=os.path.join(cfg.rl_save_path,cfg.exp_no)
        if not os.path.exists(cfg.exp_path):
            os.mkdir(cfg.exp_path)
    cfg.eval_load_path=cfg.exp_path
    cfg.rl_train=True
    cfg._init_logging_handler(cfg.mode)
    assert cfg.validate_mode in ['offline', 'online']
    cfg.turn_level_reward=True if cfg.turn_level else cfg.turn_level_reward
    #user_nlu = None
    #user_dst = None
    # rule policy
    # template NLG
    #user_nlg = TemplateNLG(is_user=True)
    # assemble
    #user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name='user')
    if cfg.joint_train:
        DS_device=cfg.cuda_device[0]
        US_device=cfg.cuda_device[1]
    else:
        DS_device=cfg.cuda_device[0]
        US_device=cfg.cuda_device[0]
    set_seed(cfg.seed)
    sys_agent = LsGPT(return_act=False, model_path=cfg.DS_path, device=DS_device)
    if cfg.rl_with_us:
        user_policy = RulePolicy(character='usr')
        user_agent=GPT2Agent(policy=user_policy, name='user_gpt', reader=sys_agent.reader, device=US_device, model_path=cfg.US_path)
    else:
        user_agent=None
    sess=RL_sess(user_agent=user_agent, sys_agent=sys_agent)
    #print('setting of non_neg_reward', cfg.non_neg_reward)
    if cfg.mode=='test':
        sess.validate_online('test')
    else:
        sess.run_RL()

