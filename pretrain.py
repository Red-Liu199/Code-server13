# supervised pretraining before RL training
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from eval import MultiWozEvaluator
#from damd_net import DAMD, cuda_, get_one_hot_input
from reader import MultiWozReader, tod_dataset, train_collate_fn, test_collate_fn
import utils
from torch.optim import Adam
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from analyze_result import prepare_for_std_eval
from mwzeval.metrics import Evaluator

import os
import shutil
import random
import argparse
import time
import logging
import json
import tqdm
import numpy as np
from compute_joint_acc import compute_jacc
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.nn.parallel import replicate
from config import global_config as cfg
# from config21 import global_config as cfg  # global, already initialized


#import warnings
#warnings.filterwarnings("ignore")

class Modal(object):
    def __init__(self, device=[0]):
        if len(device)>1:
            self.device1=device[0]
            self.device2=device[1]
        else:
            self.device1 = device[0]
            self.device2=device[0]
        if not cfg.combine_eval:
            tokenizer_path=cfg.gpt_path
            self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
            self.reader = MultiWozReader(self.tokenizer)
            self.get_special_ids()
            # logging.info([self.sos_b_id, self.sos_a_id, self.sos_r_id, self.eos_b_id, self.eos_a_id,self.eos_r_id])

            # create model: gpt2
            single_mode=['pretrain','train','test_pos']
            if cfg.mode in single_mode:
                self.model=GPT2LMHeadModel.from_pretrained(cfg.gpt_path)
                self.model.resize_token_embeddings(len(self.tokenizer))
                if cfg.gradient_checkpoint:
                    self.model.config.gradient_checkpointing=True
                
                self.model.to(self.device1)
                self.PrioriModel=self.model
            
            elif cfg.mode=='test' or cfg.mode=='test_all':
                self.PrioriModel=GPT2LMHeadModel.from_pretrained(cfg.gpt_path)
                self.model=self.PrioriModel
                if cfg.gradient_checkpoint:
                    self.PrioriModel.config.gradient_checkpointing=True
                self.PosteriorModel=None
                self.PrioriModel.to(self.device1)
            
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.gpt_path1)
            self.reader = MultiWozReader(self.tokenizer)
            self.get_special_ids()
            self.dst_model=GPT2LMHeadModel.from_pretrained(cfg.gpt_path1)
            self.dm_model=GPT2LMHeadModel.from_pretrained(cfg.gpt_path2)
            #self.nlg_model=GPT2LMHeadModel.from_pretrained(cfg.gpt_path3)
            self.dst_model.to(self.device1)
            self.dm_model.to(self.device1)
            #self.nlg_model.to(self.device1)
        self.evaluator = MultiWozEvaluator(self.reader)
        self.std_evaluator=Evaluator(bleu=1, success=1, richness=0)
        if cfg.save_log:
            log_path='./log21/log_{}'.format(cfg.exp_no) if cfg.dataset==1 else './log/log_{}'.format(cfg.exp_no)
            if os.path.exists(log_path):
                shutil.rmtree(log_path)
                os.mkdir(log_path)
            else:
                os.mkdir(log_path)
            self.tb_writer = SummaryWriter(log_dir=log_path)
        else:
            self.tb_writer = None
        cfg.origin_batch_size=cfg.batch_size
        self.eps=1e-45
        if 'test' not in cfg.mode:
            json.dump(cfg.__dict__,open(os.path.join(cfg.exp_path,'cfg_all.json'),'w'),indent=2)
        self.global_output=4

    def get_special_ids(self):
        self.sos_b_id=self.tokenizer.convert_tokens_to_ids('<sos_b>')
        self.sos_a_id=self.tokenizer.convert_tokens_to_ids('<sos_a>')
        self.sos_r_id=self.tokenizer.convert_tokens_to_ids('<sos_r>')
        self.eos_b_id=self.tokenizer.convert_tokens_to_ids('<eos_b>')
        self.eos_a_id=self.tokenizer.convert_tokens_to_ids('<eos_a>')
        self.eos_r_id=self.tokenizer.convert_tokens_to_ids('<eos_r>')
        self.sos_db_id=self.tokenizer.convert_tokens_to_ids('<sos_db>')
        self.eos_db_id=self.tokenizer.convert_tokens_to_ids('<eos_db>')
        self.sos_u_id=self.tokenizer.convert_tokens_to_ids('<sos_u>')
        self.eos_u_id=self.tokenizer.convert_tokens_to_ids('<eos_u>')
        if cfg.train_us:
            self.sos_g_id=self.tokenizer.convert_tokens_to_ids('<sos_g>')
            self.eos_g_id=self.tokenizer.convert_tokens_to_ids('<eos_g>')
            self.sos_ua_id=self.tokenizer.convert_tokens_to_ids('<sos_ua>')
            self.eos_ua_id=self.tokenizer.convert_tokens_to_ids('<eos_ua>')

    def pretrain(self, posterior=False):
        #logging.info(cfg.mode)
        if cfg.mode=='train':
            num_dials=len(self.reader.train)
            all_batches = self.reader.get_batches('train')
        else:
            #divide the datasets
            if not os.path.exists(cfg.divided_path):
                train_data=self.reader.train
                random.shuffle(train_data)
                bound=int(len(train_data)*int(cfg.spv_proportion)/100)
                self.pre_data=train_data[:bound]
                self.post_data=train_data[bound:]
                encoded_data={'pre_data':self.pre_data,'post_data':self.post_data}
                logging.info('Divided data saved in %s'%cfg.divided_path)
                json.dump(encoded_data, open(cfg.divided_path, 'w'), indent=2)
            else:
                encoded_data = json.loads(open(cfg.divided_path, 'r', encoding='utf-8').read())
                self.pre_data=encoded_data['pre_data']
                num_dials=len(self.pre_data)
            all_batches = self.reader.get_batches('train',data=self.pre_data)
            num_dials=len(self.pre_data)

        optimizer, scheduler = self.get_sep_optimizers(num_dials,self.model)

        # log info
        logging.info("***** Running pretraining *****")
        logging.info("  Num Dialogs = %d", num_dials)
        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)


        log_inputs = 2
        global_step = 0
        min_loss = 1000
        min_eval_loss=1000
        max_score=0
        early_stop_count=cfg.early_stop_count
        if cfg.use_scheduler:
            warmup_epochs=cfg.warmup_steps*cfg.gradient_accumulation_steps*cfg.batch_size//num_dials \
                if cfg.warmup_steps>=0 else int(cfg.epoch_num*cfg.warmup_ratio)
            logging.info('warmup epochs:{}'.format(warmup_epochs))
        #eval_loss=self.eval(posterior=posterior,model=self.model)
        #logging.info('initial evaluation loss:%f'%eval_loss)
        num_batch=len(all_batches)
        epoch_th=0.1*cfg.epoch_num if 'distilgpt2' in cfg.gpt_path else -1
        #epoch_th=-1
        for epoch in tqdm(range(cfg.epoch_num)):
            epoch_step = 0
            total_loss=0
            logging_loss=0
            btm = time.time()
            oom_time = 0
            self.model.zero_grad()
            #shuffle batch instead of data
            random.shuffle(all_batches)
            for batch_idx, dial_batch in enumerate(all_batches):
                if cfg.train_us:
                    inputs, labels = self.reader.convert_us_batch_session(dial_batch)
                else:
                    inputs, labels = self.reader.convert_batch_session(dial_batch,posterior_train=posterior)
                try:  # avoid OOM
                    self.model.train()
                    if log_inputs > 0 and cfg.example_log:  # log inputs for the very first two turns
                        logging.info('Input examples:')
                        logging.info(self.tokenizer.decode(inputs['contexts'][0]))
                        log_inputs-=1
                    

                    # to tensor
                    inputs = self.add_torch_input(inputs,posterior=posterior)#B,T
                    labels=self.add_torch_input(labels,posterior=posterior)#B,T
                    
                    # loss
                    outputs = self.model(inputs['contexts_tensor'])
                    if cfg.only_target_loss:
                        loss=self.calculate_loss_and_accuracy(outputs,labels['contexts_tensor'])
                    else:
                        loss=self.calculate_loss_and_accuracy(outputs,inputs['contexts_tensor'])
                    if cfg.loss_reg:
                        loss=loss/cfg.gradient_accumulation_steps
                    loss.backward()
                    total_loss+=loss.item()
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    epoch_step += 1

                    if (batch_idx+1) % cfg.gradient_accumulation_steps == 0 or((batch_idx + 1) == num_batch):
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1

                        loss_scalar = (total_loss - logging_loss) / cfg.gradient_accumulation_steps
                        
                        logging_loss = total_loss
                        
                        if self.tb_writer:
                            self.tb_writer.add_scalar('lr', optimizer.param_groups[0]["lr"],global_step)
                            self.tb_writer.add_scalar('loss', loss_scalar, global_step)
                            

                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        max_length = max(inputs['lengths'])
                        oom_time += 1
                        logging.info("WARNING: ran out of memory,times: {}, batch size: {}, max_len: {}".format(
                            oom_time, cfg.batch_size, max_length))
                        if hasattr(torch.cuda, 'empty_cache'):
                            with torch.cuda.device(self.model.device):
                                torch.cuda.empty_cache()
                    else:
                        logging.info(str(exception))
                        raise exception
            logging.info('Epoch:{}, Train epoch time: {:.2f} min, loss: {}'.format(
                epoch, (time.time()-btm)/60, total_loss/epoch_step))
            
            if cfg.evaluate_during_training:
                if cfg.save_type=='min_loss':
                    eval_loss=self.eval(model=self.model,posterior=posterior)
                    logging.info('model evaluation loss:{}'.format(eval_loss))
                    if self.tb_writer:
                        self.tb_writer.add_scalar('loss_eval',eval_loss,epoch)
                    if eval_loss<min_eval_loss:
                        min_eval_loss=eval_loss
                        self.save_model(path='best_loss_model',model=self.model)
                        early_stop_count=cfg.early_stop_count
                    else:
                        if epoch>=warmup_epochs:#early stop after warm up
                            early_stop_count-=1
                            logging.info('early stop count:%d'%early_stop_count)
                            if early_stop_count==0 and cfg.early_stop:
                                logging.info('early stopped')
                                break
                elif cfg.save_type=='max_score' and epoch>epoch_th:
                    if posterior:
                        eval_result=self.validate_pos(data='dev')
                        self.tb_writer.add_scalar('joint_goal',eval_result['joint_acc'],epoch)
                        self.tb_writer.add_scalar('act_F1',eval_result['act_F1'],epoch)
                        self.tb_writer.add_scalar('db_acc',eval_result['db_acc'],epoch)
                        score=eval_result['joint_acc']
                    else:
                        eval_result=self.validate_fast(data='dev')
                        self.tb_writer.add_scalar('joint_goal',eval_result['joint_acc'],epoch)
                        self.tb_writer.add_scalar('match',eval_result['match'],epoch)
                        self.tb_writer.add_scalar('success',eval_result['success'],epoch)
                        self.tb_writer.add_scalar('bleu',eval_result['bleu'],epoch)
                        self.tb_writer.add_scalar('combined_score',eval_result['score'],epoch)
                        score=eval_result['score']
                    if score>max_score:
                        early_stop_count=cfg.early_stop_count
                        max_score=score
                        self.save_model(path='best_score_model',model=self.model)
                    else:
                        if epoch>=warmup_epochs:
                            early_stop_count-=1
                            logging.info('early stop count:%d'%early_stop_count)
                            if early_stop_count==0 and cfg.early_stop:
                                logging.info('early stopped')
                                break
            else:#save the model with minimal training loss
                if total_loss/epoch_step<min_loss:
                    min_loss=total_loss/epoch_step
                    self.save_model(posterior=posterior,model=self.model)

    def pretrain_turn_level(self):
        if cfg.mode=='train':
            num_dials=len(self.reader.train)
            all_batches = self.reader.get_batches('train')
        else:
            #divide the datasets
            if not os.path.exists(cfg.divided_path):
                train_data=self.reader.train
                random.shuffle(train_data)
                bound=int(len(train_data)*int(cfg.spv_proportion)/100)
                self.pre_data=train_data[:bound]
                self.post_data=train_data[bound:]
                encoded_data={'pre_data':self.pre_data,'post_data':self.post_data}
                logging.info('Divided data saved in %s'%cfg.divided_path)
                json.dump(encoded_data, open(cfg.divided_path, 'w'), indent=2)
            else:
                encoded_data = json.loads(open(cfg.divided_path, 'r', encoding='utf-8').read())
                self.pre_data=encoded_data['pre_data']
                num_dials=len(self.pre_data)
            all_batches = self.reader.get_batches('train',data=self.pre_data)
            num_dials=len(self.pre_data)
        set_stats = self.reader.set_stats['train']
        num_turns=set_stats['num_turns']
        optimizer, scheduler = self.get_sep_optimizers(num_turns,self.model)

        # log info
        logging.info("***** Running turn-level training *****")
        logging.info("  Num Training steps(one turn in a batch of dialogs) per epoch = %d",
                     set_stats['num_training_steps_per_epoch'])
        logging.info("  Num Turns = %d", set_stats['num_turns'])
        logging.info("  Num Dialogs = %d", set_stats['num_dials'])
        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)
        logging.info("  Gradient Accumulation steps = %d",
                     cfg.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d",
                     set_stats['num_training_steps_per_epoch']*cfg.epoch_num // cfg.gradient_accumulation_steps)


        log_inputs = 4
        global_step = 0
        sw = time.time()

        min_loss = 1000
        min_eval_loss=1000
        max_score=0
        early_stop_count=cfg.early_stop_count
        epoch_th=0.1*cfg.epoch_num if 'distilgpt2' in cfg.gpt_path else -1
        #epoch_th=-1
        warmup_epochs=cfg.warmup_steps*cfg.gradient_accumulation_steps*cfg.batch_size//num_dials \
            if cfg.warmup_steps>=0 else int(cfg.epoch_num*cfg.warmup_ratio)
        if cfg.debugging:
            self.validate_fast()
        for epoch in range(cfg.epoch_num):
            epoch_step = 0
            tr_loss = 0.0
            logging_loss = 0.0
            btm = time.time()
            oom_time = 0
            self.model.zero_grad()
            random.shuffle(all_batches)
            data_iterator = self.reader.get_data_iterator(all_batches)

            for batch_idx, dial_batch in enumerate(data_iterator):
                pv_batch = None
                for turn_num, turn_batch in enumerate(dial_batch):
                    first_turn = (turn_num == 0)
                    side='user' if cfg.train_us else 'sys'
                    inputs, labels = self.reader.convert_batch_turn(turn_batch, pv_batch, first_turn, side=side)
                    if cfg.train_us:
                        pv_batch = self.reader.get_pv_batch(pv_batch, resp=turn_batch['resp'], 
                            aspn=turn_batch['sys_act'], side=side)
                    else:
                        pv_batch = self.reader.get_pv_batch(pv_batch, user=turn_batch['user'],
                            resp=turn_batch['resp'], bspn=turn_batch['bspn'], side=side)
                    try:  # avoid OOM
                        self.model.train()
                        if log_inputs > 0:  # log inputs for the very first two turns
                            logging.info('Input examples:')
                            logging.info(self.tokenizer.decode(inputs['contexts'][0]))
                            log_inputs-=1

                        # to tensor
                        inputs = self.add_torch_input(inputs)
                        # loss
                        outputs = self.model(inputs['contexts_tensor'])
                        if cfg.only_target_loss:
                            labels=self.add_torch_input(labels)    
                            loss = self.calculate_loss_and_accuracy(outputs, labels=labels['contexts_tensor'])
                        else:
                            loss = self.calculate_loss_and_accuracy(outputs, labels=inputs['contexts_tensor'])
                        if cfg.loss_reg:
                            loss=loss/cfg.gradient_accumulation_steps
                        loss.backward()
                        tr_loss += loss.item()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 5.0)
                        epoch_step += 1

                        # step, wrt gradient_accumulation_steps, clip grad norm
                        if (epoch_step+1) % cfg.gradient_accumulation_steps == 0 or(
                            # end of an epoch
                            (epoch_step + \
                            1) == set_stats['num_training_steps_per_epoch']
                        ):
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                            # global_step: actual step the optimizer took
                            global_step += 1

                    except RuntimeError as exception:
                        if "out of memory" in str(exception):
                            max_length = max(inputs['lengths'])
                            oom_time += 1
                            logging.info("WARNING: ran out of memory,times: {}, batch size: {}, max_len: {}".format(
                                oom_time, cfg.batch_size, max_length))
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                        else:
                            logging.info(str(exception))
                            raise exception
            logging.info('Epoch:{}, Train epoch time: {:.2f} min, epoch loss: {:.4f}'.format(
                epoch, (time.time()-btm)/60, tr_loss))
            if cfg.evaluate_during_training:
                if cfg.save_type=='min_loss':
                    eval_loss=self.eval(model=self.model,posterior=posterior)
                    logging.info('model evaluation loss:{}'.format(eval_loss))
                    if self.tb_writer:
                        self.tb_writer.add_scalar('loss_eval',eval_loss,epoch)
                    if eval_loss<min_eval_loss:
                        min_eval_loss=eval_loss
                        self.save_model(path='best_loss_model',model=self.model)
                        early_stop_count=cfg.early_stop_count
                    else:
                        if epoch>=warmup_epochs:#early stop after warm up
                            early_stop_count-=1
                            logging.info('early stop count:%d'%early_stop_count)
                            if early_stop_count==0 and cfg.early_stop:
                                logging.info('early stopped')
                                break
                elif cfg.save_type=='max_score' and epoch>epoch_th:
                    if cfg.train_us:
                        bleu, P, R, F1=self.validate_us(data='dev')
                        self.tb_writer.add_scalar('P',P,epoch)
                        self.tb_writer.add_scalar('R',R,epoch)
                        self.tb_writer.add_scalar('F1',F1,epoch)
                        self.tb_writer.add_scalar('bleu',bleu,epoch)
                        score=F1*100
                    else:
                        eval_result=self.validate_fast(data='dev')
                        self.tb_writer.add_scalar('joint_goal',eval_result['joint_acc'],epoch)
                        self.tb_writer.add_scalar('match',eval_result['match'],epoch)
                        self.tb_writer.add_scalar('success',eval_result['success'],epoch)
                        self.tb_writer.add_scalar('bleu',eval_result['bleu'],epoch)
                        self.tb_writer.add_scalar('combined_score',eval_result['score'],epoch)
                        score=eval_result['score']
                    if score>max_score:
                        early_stop_count=cfg.early_stop_count
                        max_score=score
                        self.save_model(path='best_score_model',model=self.model)
                    else:
                        if epoch>=warmup_epochs:
                            early_stop_count-=1
                            logging.info('early stop count:%d'%early_stop_count)
                            if early_stop_count==0 and cfg.early_stop:
                                logging.info('early stopped')
                                break
            
    
    def pretrain_sys_model(self):
        data=self.reader.train
        all_batches, seq_num = self.reader.get_sys_batch(data, batch_size=cfg.batch_size)
        num_batch=len(all_batches)
        logging.info('Total batches:{}'.format(num_batch))
        optimizer, scheduler = self.get_sep_optimizers(seq_num,self.model)
        training_steps=0
        min_eval_loss=1000
        early_stop_count=cfg.early_stop_count
        for epoch in range(cfg.epoch_num):
            random.shuffle(all_batches)
            self.model.train()
            st=time.time()
            for batch_idx, batch in enumerate(all_batches):
                if self.global_output>0:
                    logging.info('Training examples:')
                    logging.info(self.tokenizer.decode(list(batch)[0]))
                    self.global_output-=1
                input_batch=torch.from_numpy(batch).long().to(self.model.device)
                output_batch=self.model(input_batch)
                loss = self.calculate_loss_and_accuracy(output_batch, input_batch)
                loss.backward()
                self.tb_writer.add_scalar('training_loss', loss.item(), training_steps)
                training_steps+=1
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                if (batch_idx+1) % cfg.gradient_accumulation_steps == 0 or((batch_idx + 1) == num_batch):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            logging.info('Epoch:{}, Train epoch time: {:.2f} min'.format(epoch, (time.time()-st)/60))
            eval_loss=self.eval_sys()
            self.tb_writer.add_scalar('eval_loss',eval_loss,epoch)
            if eval_loss<min_eval_loss:
                min_eval_loss=eval_loss
                self.save_model(path='best_loss_model',model=self.model)
                early_stop_count=cfg.early_stop_count
            else:
                early_stop_count-=1
                logging.info('early stop count:%d'%early_stop_count)
                if early_stop_count==0 and cfg.early_stop:
                    logging.info('early stopped')
                    break

    def train_modular(self, modular='dst'):
        data_path='data/multi-woz-2.1-processed/data_for_{}.json'.format(modular)
        encoded_path='data/multi-woz-2.1-processed/encoded_data_for_{}.json'.format(modular)
        if os.path.exists(encoded_path):
            encoded_data=json.loads(open(encoded_path, 'r', encoding='utf-8').read())
            logging.info('Reading encoded data from {}'.format(encoded_path))
        else:
            data=json.loads(open(data_path, 'r', encoding='utf-8').read())
            logging.info('Starting encoding data')
            st=time.time()
            encoded_data=self.reader.encode_data(data, self.tokenizer, modular=modular)
            logging.info('Encoding time:{:.3f} min. Encoded data saved in {}'.format((time.time()-st)/60, encoded_path))
            json.dump(encoded_data, open(encoded_path, 'w'))
        
        train_dataloader=DataLoader(encoded_data['train'], batch_size=cfg.batch_size, shuffle=True, collate_fn=train_collate_fn)
        dev_dataloader=DataLoader(encoded_data['dev'], batch_size=cfg.eval_batch_size, collate_fn=test_collate_fn)
        num_turns=len(encoded_data['train'])
        logging.info('Num turns = {}'.format(num_turns))
        optimizer, scheduler = self.get_sep_optimizers(num_turns,self.model)

        log_inputs = 4
        global_step = 0
        max_score=0
        early_stop_count=cfg.early_stop_count
        epoch_th=0.05*cfg.epoch_num if 'distilgpt2' in cfg.gpt_path else -1
        #epoch_th=-1
        warmup_epochs=cfg.warmup_steps*cfg.gradient_accumulation_steps*cfg.batch_size//num_turns \
            if cfg.warmup_steps>=0 else int(cfg.epoch_num*cfg.warmup_ratio)
        logging.info('Warmup epochs:{}'.format(warmup_epochs))

        for epoch in range(cfg.epoch_num):
            tr_loss = 0.0
            step_loss=0
            btm = time.time()
            oom_time = 0
            self.model.zero_grad()

            for batch_idx, batch in enumerate(train_dataloader):
                try:  # avoid OOM
                    self.model.train()
                    inputs=batch[0].to(self.model.device)#B, T
                    if log_inputs > 0:  # log inputs for the very first two turns
                        logging.info('Input examples:')
                        logging.info(self.tokenizer.decode(inputs[0,:]))
                        log_inputs-=1
                    labels=batch[1].to(self.model.device) if cfg.only_target_loss else inputs
                    outputs = self.model(inputs)
                    loss = self.calculate_loss_and_accuracy(outputs, labels=labels)
                    if cfg.loss_reg:
                        loss=loss/cfg.gradient_accumulation_steps
                    loss.backward()
                    tr_loss += loss.item()
                    step_loss+=loss.item()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    if (batch_idx+1) % cfg.gradient_accumulation_steps == 0 or batch_idx+1==len(train_dataloader):
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1
                        self.tb_writer.add_scalar('lr', optimizer.param_groups[0]["lr"], global_step)
                        self.tb_writer.add_scalar('loss', step_loss, global_step)
                        step_loss=0

                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        oom_time += 1
                        logging.info("WARNING: ran out of memory,times: {}, batch size: {}".format(oom_time, cfg.batch_size))
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        logging.info(str(exception))
                        raise exception
            logging.info('Epoch:{}, Train epoch time: {:.2f} min, epoch loss: {:.4f}'.format(
                epoch, (time.time()-btm)/60, tr_loss))
            
            if epoch>epoch_th:
                gen=[]
                oracle=[]
                if modular=='dst':
                    max_len=60
                    sos_id=self.sos_b_id
                    eos_id=self.eos_b_id
                elif modular=='nlg':
                    max_len=60
                    sos_id=self.sos_r_id
                    eos_id=self.eos_r_id
                elif modular=='dm':
                    max_len=25
                    sos_id=self.sos_a_id
                    eos_id=self.eos_a_id
                st=time.time()
                for batch in dev_dataloader:
                    gen_batch=self.generate_batch(self.model, batch[0], max_len, eos_id)
                    gen+=self.convert_batch_ids_to_tokens(self.tokenizer, gen_batch, sos_id, eos_id)
                    oracle+=self.convert_batch_ids_to_tokens(self.tokenizer, batch[1], sos_id, eos_id)
                logging.info('Validation time:{:.2f} min'.format((time.time()-st)/60))
                metrics=self.evaluator.calculate_metrics(gen, oracle, modular=modular)
                if modular in ['dst', 'dm']:
                    logging.info('Dev joint accuracy:{:.3f}, P/R/F1:{}'.format(metrics[0], metrics[1]))
                    score=metrics[1][2] # F1 score
                    self.tb_writer.add_scalar('joint_acc', metrics[0], epoch)
                    self.tb_writer.add_scalar('precious', metrics[1][0], epoch)
                    self.tb_writer.add_scalar('recall', metrics[1][1], epoch)
                    self.tb_writer.add_scalar('F1', metrics[1][2], epoch)
                else:
                    logging.info('Dev BLEU:{:.3f}'.format(metrics))
                    score=metrics #BLEU
                    self.tb_writer.add_scalar('BLEU', metrics, epoch)

                if score>max_score:
                    early_stop_count=cfg.early_stop_count
                    max_score=score
                    self.save_model(path='best_score_model',model=self.model)
                else:
                    if epoch>=warmup_epochs:
                        early_stop_count-=1
                        logging.info('early stop count:%d'%early_stop_count)
                        if early_stop_count==0 and cfg.early_stop:
                            logging.info('early stopped')
                            break

    def eval_modular(self, modular='dst'):
        if modular=='dst':
            self.eval_dst()
        data_path='data/multi-woz-2.1-processed/data_for_{}.json'.format(modular)
        data=json.loads(open(data_path, 'r', encoding='utf-8').read())
        encoded_path='data/multi-woz-2.1-processed/encoded_data_for_{}.json'.format(modular)
        encoded_data=json.loads(open(encoded_path, 'r', encoding='utf-8').read())
        logging.info('Total evaluation turns:{}'.format(len(encoded_data['test'])))
        test_dataloader=DataLoader(encoded_data['test'], batch_size=cfg.eval_batch_size, collate_fn=test_collate_fn)
        gen=[]
        oracle=[]
        if modular=='dst':
            max_len=60
            sos_id=self.sos_b_id
            eos_id=self.eos_b_id
        elif modular=='nlg':
            max_len=60
            sos_id=self.sos_r_id
            eos_id=self.eos_r_id
        elif modular=='dm':
            max_len=25
            sos_id=self.sos_a_id
            eos_id=self.eos_a_id
        st=time.time()
        for batch in test_dataloader:
            gen_batch=self.generate_batch(self.model, batch[0], max_len, eos_id)
            gen+=self.convert_batch_ids_to_tokens(self.tokenizer, gen_batch, sos_id, eos_id)
            oracle+=self.convert_batch_ids_to_tokens(self.tokenizer, batch[1], sos_id, eos_id)
        results=[{'context':a[0], 'oracle':b, 'gen':c} for (a, b, c) in zip(data['test'], oracle, gen)]
        json.dump(results, open(os.path.join(cfg.eval_load_path, 'result.json'), 'w'), indent=2)
        logging.info('Validation time:{:.2f} min'.format((time.time()-st)/60))
        metrics=self.evaluator.calculate_metrics(gen, oracle, modular=modular)
        if modular in ['dst', 'dm']:
            logging.info('Dev joint accuracy:{:.3f}, P/R/F1:{}'.format(metrics[0], metrics[1]))
        else:
            logging.info('Dev BLEU:{:.3f}'.format(metrics))

    def eval_dst(self):
        cfg.batch_size=cfg.eval_batch_size
        batches=self.reader.get_batches('test')
        self.model.eval()
        gen=[]
        oracle=[]
        results=[]
        turn_results=[]
        log_output=3
        with torch.no_grad():
            for batch in batches:
                batch=self.reader.transpose_batch(batch)
                pv_bspn=None
                for turn_num, turn_batch in enumerate(batch):
                    contexts=self.reader.convert_dst_eval_batch(turn_batch, pv_bspn=pv_bspn)
                    gen_batch_ids=self.generate_batch(self.model, contexts, max_len=60, eos_id=self.eos_b_id)
                    gen_bspn_batch, pv_bspn=self.convert_batch_ids_to_tokens(self.tokenizer, gen_batch_ids, self.sos_b_id, self.eos_b_id, return_ids=True)
                    oracle_bspn_batch=self.convert_batch_ids_to_tokens(self.tokenizer, turn_batch['bspn'], self.sos_b_id, self.eos_b_id)
                    gen+=gen_bspn_batch
                    oracle+=oracle_bspn_batch
                    turn_results+=[{'bspn':a, 'bspn_gen':b} for a, b in zip(oracle_bspn_batch, gen_bspn_batch)]
                    results+=[{'context':self.tokenizer.decode(a), 'oracle':b, 'gen':c} for (a, b, c) in zip(contexts, oracle_bspn_batch, gen_bspn_batch)]
                    if log_output>0:
                        log_output-=1
                        logging.info('Context:{}\nGen:{}\nOracle:{}'.format(self.tokenizer.decode(contexts[3]), gen_bspn_batch[3], oracle_bspn_batch[3]))
        metrics=self.evaluator.calculate_metrics(gen, oracle, modular='dst')
        logging.info('(Standard) Dev joint accuracy:{:.3f}, P/R/F1:{}'.format(metrics[0], metrics[1]))
        json.dump(results, open(os.path.join(cfg.eval_load_path, 'std_result.json'), 'w'), indent=2)
        joint_acc=compute_jacc(turn_results)
        logging.info('Previous joint acc:{:.3f}'.format(joint_acc))

    def combine_modules(self):
        cfg.batch_size=cfg.eval_batch_size
        batches=self.reader.get_batches('test')
        self.dst_model.eval()
        self.dm_model.eval()
        #self.nlg_model.eval()
        result_collection={}
        st=time.time()
        with torch.no_grad():
            for batch in batches:
                new_batch=[]
                batch_size=len(batch)
                batch=self.reader.transpose_batch(batch)
                pv_bspn_ids=None
                pv_resp_ids=None
                pv_bspn=None
                self.turn_domain_batch=['' for _ in range(batch_size)]
                for turn_num, turn_batch in enumerate(batch):
                    contexts=self.reader.convert_dst_eval_batch(turn_batch, pv_bspn=pv_bspn_ids, pv_resp=pv_resp_ids)
                    gen_batch_ids=self.generate_batch(self.dst_model, contexts, max_len=60, eos_id=self.eos_b_id)
                    gen_bspn, gen_bspn_ids=self.convert_batch_ids_to_tokens(self.tokenizer, gen_batch_ids, self.sos_b_id, self.eos_b_id, return_ids=True)
                    db_batch, db_batch_ids=self.get_db_batch(gen_bspn, pv_bspn, return_ids=True)

                    contexts=self.reader.convert_dm_eval_batch(turn_batch, gen_bspn_ids, db_batch_ids, pv_bspn_ids, pv_resp_ids)
                    gen_batch_ids=self.generate_batch(self.dm_model, contexts, max_len=20, eos_id=self.eos_a_id)
                    gen_aspn, gen_aspn_ids=self.convert_batch_ids_to_tokens(self.tokenizer, gen_batch_ids, self.sos_a_id, self.eos_a_id, return_ids=True)
                    
                    contexts=self.reader.convert_nlg_eval_batch(turn_batch, gen_bspn_ids, db_batch_ids, gen_aspn_ids, pv_bspn_ids, pv_resp_ids)
                    gen_batch_ids=self.generate_batch(self.dm_model, contexts, max_len=60, eos_id=self.eos_r_id)
                    gen_resp, gen_resp_ids=self.convert_batch_ids_to_tokens(self.tokenizer, gen_batch_ids, self.sos_r_id, self.eos_r_id, return_ids=True)

                    pv_bspn_ids=gen_bspn_ids
                    pv_resp_ids=gen_resp_ids
                    pv_bspn=gen_bspn
                    turn_batch['bspn_gen']=gen_bspn_ids
                    turn_batch['aspn_gen']=gen_aspn_ids
                    turn_batch['resp_gen']=gen_resp_ids
                    turn_batch['db_gen']=db_batch_ids
                    new_batch.append(turn_batch)
                batch=self.reader.inverse_transpose_batch(new_batch)
                for dialog in batch:
                    result_collection.update(self.reader.inverse_transpose_turn(dialog))
        results, field = self.reader.wrap_result_lm(result_collection)
        logging.info('Inference time:{:.3f} min'.format((time.time()-st)/60))
        
        joint_acc=compute_jacc(results)
        cfg.use_true_bspn_for_ctr_eval=False
        bleu, success, match = self.evaluator.validation_metric(results)
        score = 0.5 * (success + match) + bleu
        logging.info('validation %2.2f  %2.2f  %2.2f  %.2f  %.3f' % (match, success, bleu, score, joint_acc))
        input_data=prepare_for_std_eval(data=results)
        std_metrics = self.std_evaluator.evaluate(input_data)
        bleu=std_metrics['bleu']['damd']
        match=std_metrics['success']['inform']['total']
        success=std_metrics['success']['success']['total']
        score = 0.5 * (success + match) + bleu
        if cfg.mode=='test':
            logging.info(std_metrics)
        logging.info('[Std] validation %2.2f  %2.2f  %2.2f  %.2f  %.3f' % (match, success, bleu, score, joint_acc))
        json.dump(results, open(os.path.join(cfg.gpt_path1, 'result.json'), 'w'), indent=2)


    def get_db_batch(self, bs_batch, pv_bs_batch=None, return_ids=False):
        batch_size=len(bs_batch)
        db_batch=[]
        db_batch_ids=[]
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
            if return_ids:
                db_batch_ids.append(self.tokenizer.encode(db_result))
        if return_ids:
            return db_batch, db_batch_ids
        return db_batch

    def convert_batch_ids_to_tokens(self, tokenizer, input_ids, sos_id, eos_id, return_ids=False):
        # input_ids: B*T
        # output: B*string
        outputs=[]
        outputs_ids=[]
        for sent_ids in input_ids:
            if eos_id in sent_ids:
                sent_ids=sent_ids[:sent_ids.index(eos_id)+1]
            else:
                sent_ids[-1]=eos_id
            if sos_id not in sent_ids:
                sent_ids=[sos_id]+sent_ids
            outputs_ids.append(sent_ids)
            outputs.append(tokenizer.decode(sent_ids))
        if return_ids:
            return outputs, outputs_ids
        return outputs

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

    def add_torch_input(self, inputs, posterior=False):
        # to tensor and to device
        contexts_tensor = torch.from_numpy(inputs['contexts_np']).long()
        if posterior:
            contexts_tensor = contexts_tensor.to(self.device2)
        else:
            contexts_tensor = contexts_tensor.to(self.device1)
        inputs['contexts_tensor'] = contexts_tensor
        return inputs


    def calculate_loss_and_accuracy(self, outputs, labels):
        # GPT2-chicahat/train.py
        lm_logits = outputs[0]

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        pad_id = cfg.pad_id
        loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # avg loss
        not_ignore = shift_labels.ne(pad_id)
        num_targets = not_ignore.long().sum().item()

        loss /= num_targets
        return loss


    def convert_eval_batch(self,data,contexts, turn_num,bs_gen,prior=False,db_gen=None,resp_gen=None,aspn_gen=None, gen_db=False):
        
        if gen_db:#???????????????????????????????????????????????????
            new_contexts=[]
            for id, context in enumerate(contexts):
                new_contexts.append(context[:-1] + bs_gen[id] + [self.sos_db_id])
            return new_contexts
        else:
            for id,context in enumerate(contexts):
                if turn_num==0:
                    if prior:
                        if db_gen is None:#???????????????bs_gen???????????????db_gen
                            contexts[id]=data[id][turn_num]['user']+[self.sos_b_id]
                        else:
                            sos_id=self.sos_a_id if cfg.model_act else self.sos_r_id
                            contexts[id]=context[:-1]+bs_gen[id]+db_gen[id]+[sos_id]
                    else:
                        if db_gen is None:
                            contexts[id]=data[id][turn_num]['user']+data[id][turn_num]['resp']+[self.sos_b_id]
                        else:
                            contexts[id]= context[:-1] + bs_gen[id]+ db_gen[id] + [self.sos_a_id]
                else:
                    #context???????????????sos_b???
                    if prior:
                        if resp_gen is None:
                            sos_id=self.sos_a_id if cfg.model_act else self.sos_r_id
                            contexts[id]=context[:-1] +bs_gen[id]+db_gen[id]+[sos_id]
                        else:
                            contexts[id]=context[:-1] + resp_gen[id] + data[id][turn_num]['user']+[self.sos_b_id]
                    else:
                        if resp_gen is None:
                            if cfg.model_act:
                                contexts[id]=context[:-1] + bs_gen[id] + db_gen[id] + [self.sos_a_id]#to generate aspn
                            else:
                                contexts[id]=context[:-1] + bs_gen[id] +[self.sos_r_id]
                        else:
                            if cfg.model_act:
                                contexts[id]=context[:-1] + aspn_gen[id] + data[id][turn_num]['user']\
                                    +data[id][turn_num]['resp']+[self.sos_b_id]#to generate bspn
                            else:
                                contexts[id]=context[:-1]+data[id][turn_num]['user']+data[id][turn_num]['resp']+[self.sos_b_id]
            return contexts

    def get_bspn(self,bs_tensor,return_db=False,data=None,turn_num=None):
        #return db, data and turn_num must be together
        #data represents one batch
        bs_batch=bs_tensor.cpu().tolist()
        bs_gen=[]
        db_gen=[]
        eos_b_id=self.eos_b_id
        sos_b_id=self.sos_b_id
        for i,bs in enumerate(bs_batch):
            if eos_b_id in bs:
                bs=[sos_b_id]+bs[:bs.index(eos_b_id)+1]
            else:
                bs[-1]=eos_b_id
                bs=[sos_b_id]+bs
            if bs.count(sos_b_id)>1:
                last=bs[::-1].index(sos_b_id)+1
                bs=bs[-last:]

            bs_gen.append(bs)
            if return_db:
                if cfg.turn_level:
                    db_result=self.reader.bspan_to_DBpointer(self.tokenizer.decode(bs), data[turn_num]['turn_domain'][i])
                else:
                    db_result=self.reader.bspan_to_DBpointer(self.tokenizer.decode(bs), data[i][turn_num]['turn_domain'])
                db = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('<sos_db> '+ db_result + ' <eos_db>'))
                db_gen.append(db)
        if return_db:
            return bs_gen,db_gen
        else:
            return bs_gen

    def get_aspn(self,aspn_tensor):
        aspn_batch=aspn_tensor.cpu().tolist()
        aspn_gen=[]
        eos_a_id=self.eos_a_id
        sos_a_id=self.sos_a_id
        for i ,aspn in enumerate(aspn_batch):
            if eos_a_id in aspn:
                aspn=[sos_a_id]+aspn[:aspn.index(eos_a_id)+1]
            else:
                aspn[-1]=eos_a_id
                aspn=[sos_a_id]+aspn
            if aspn.count(sos_a_id)>1:
                last=aspn[::-1].index(sos_a_id)+1
                aspn=aspn[-last:]
            aspn_gen.append(aspn)
        return aspn_gen

    def get_user(self,u_tensor):
        u_batch=u_tensor.cpu().tolist()
        u_gen=[]

        for i ,u in enumerate(u_batch):
            if self.eos_u_id in u:
                u=[self.sos_ua_id]+u[:u.index(self.eos_u_id)+1]
            else:
                u[-1]=self.eos_u_id
                u=[self.sos_ua_id]+u
            if u.count(self.sos_ua_id)>1:
                last=u[::-1].index(self.sos_ua_id)+1
                u=u[-last:]
            u_gen.append(u)
        return u_gen

    def get_resp(self,resp_tensor):
        resp_batch=resp_tensor.cpu().tolist()
        resp_gen=[]
        eos_r_id=self.eos_r_id
        sos_r_id=self.sos_a_id if cfg.model_act else self.sos_r_id
        for i,resp in enumerate(resp_batch):
            if eos_r_id in resp:
                resp=[sos_r_id]+resp[:resp.index(eos_r_id)+1]
            else:
                resp[-1]=eos_r_id
                resp=[sos_r_id]+resp
            if resp.count(sos_r_id)>1:
                last=resp[::-1].index(sos_r_id)+1
                resp=resp[-last:]
            resp_gen.append(resp)
        return resp_gen

    def save_model(self, posterior=False, path=None, model=None):
        if not path:
            if posterior:
                save_path = os.path.join(cfg.exp_path, 'best_model_post')
            else:
                save_path = os.path.join(cfg.exp_path, 'best_model_pri')
        else:
            save_path = os.path.join(cfg.exp_path, path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logging.info('Saving model checkpoint to %s', save_path)
        if not model:
            if posterior:
                self.PosteriorModel.save_pretrained(save_path)
            else:
                self.PrioriModel.save_pretrained(save_path)
        else:
            model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        # save cfg
    
    def eval_sys(self, data='dev'):
        data=self.reader.dev if data=='dev' else self.reader.test
        all_batches, seq_num = self.reader.get_sys_batch(data, batch_size=cfg.batch_size)
        self.model.eval()
        total_loss=0
        with torch.no_grad():
            for batch in all_batches:
                input_batch=torch.from_numpy(batch).long().to(self.model.device)
                output_batch=self.model(input_batch)
                loss = self.calculate_loss_and_accuracy(output_batch, input_batch)
                total_loss+=loss.item()
        return total_loss/len(all_batches)
            

    def eval(self,data='dev', posterior=False, model=None):
        model.eval()
        temp=cfg.batch_size
        cfg.batch_size=cfg.origin_batch_size
        all_batches = self.reader.get_batches(data)
        total_batch=len(all_batches)
        total_loss=0
        with torch.no_grad():
            for batch in all_batches:
                if batch==[]:
                    continue
                if cfg.train_us:
                    inputs,labels=self.reader.convert_us_batch_session(batch)
                else:
                    inputs,labels=self.reader.convert_batch_session(batch,posterior_train=posterior)
                inputs=self.add_torch_input(inputs)#B,T
                labels=self.add_torch_input(labels)#B,T
                outputs = model(inputs['contexts_tensor'])
                loss=self.calculate_loss_and_accuracy(outputs,labels['contexts_tensor'])
                total_loss+=loss.item()
        cfg.batch_size=temp
        return total_loss/total_batch

    def validate_fast(self,data='dev'):
        # predict one dialog/ one turn at a time
        if cfg.mode=='pretrain' or cfg.mode=='train' or cfg.mode=='semi_ST':
            self.PrioriModel=self.model
            self.device1=self.model.device
        
        self.PrioriModel.eval()
        eval_data = self.reader.get_eval_data(data)
        if cfg.debugging:
            eval_data=eval_data[:32]
        cfg.batch_size=cfg.eval_batch_size
        batches=self.reader.get_batches('test',data=eval_data)
        result_path=os.path.join(cfg.eval_load_path,'result.json')
        
        if os.path.exists(result_path) and cfg.mode=='test':
            #results,field=self.reader.load_result(result_path)
            results=json.load(open(result_path, 'r'))
            joint_acc=compute_jacc(results)
            #joint_acc=0
            cfg.use_true_bspn_for_ctr_eval=False
            bleu, success, match = self.evaluator.validation_metric(results)
            score = 0.5 * (success + match) + bleu
            logging.info('[Old] validation %2.2f  %2.2f  %2.2f  %.2f  %.3f' % (match, success, bleu, score, joint_acc))
            
            input_data=prepare_for_std_eval(data=results)
            std_metrics = self.std_evaluator.evaluate(input_data)
            bleu=std_metrics['bleu']['damd']
            match=std_metrics['success']['inform']['total']
            success=std_metrics['success']['success']['total']
            score = 0.5 * (success + match) + bleu
            logging.info(std_metrics)
            logging.info('[Std] validation %2.2f  %2.2f  %2.2f  %.2f  %.3f' % (match, success, bleu, score, joint_acc))

            eval_results = {}
            eval_results['bleu'] = bleu
            eval_results['success'] = success
            eval_results['match'] = match
            eval_results['score'] = score
            eval_results['joint_acc']=joint_acc
            return eval_results
        
        # valid_losses = []
        result_collection = {}
        st=time.time()
        for batch in batches:
            try:
                if batch==[]:
                    continue
                if cfg.turn_level:
                    batch=self.generate_batch_turn_level(batch)
                else:
                    batch=self.generate_batch_e2e(batch)
                for dialog in batch:
                    result_collection.update(self.reader.inverse_transpose_turn(dialog))
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    logging.info("WARNING: ran out of memory during validation and batch will be divided by half, batch size:{}, turn num:{}"\
                        .format(len(batch),len(batch[0])))
                    if hasattr(torch.cuda, 'empty_cache'):
                        with torch.cuda.device(self.device1):
                            torch.cuda.empty_cache()
                    #divide the batch in half if out of memory
                    batches.insert(0,batch[:len(batch)//2])
                    batches.insert(1,batch[len(batch)//2:])
                else:
                    logging.info(str(exception))
                    raise exception
        results, field = self.reader.wrap_result_lm(result_collection)
        logging.info('Inference time:{:.3f} min'.format((time.time()-st)/60))

        joint_acc=compute_jacc(results)
        #joint_acc=0
        cfg.use_true_bspn_for_ctr_eval=False
        bleu, success, match = self.evaluator.validation_metric(results)
        score = 0.5 * (success + match) + bleu
        logging.info('[Old] validation %2.2f  %2.2f  %2.2f  %.2f  %.3f' % (match, success, bleu, score, joint_acc))
        input_data=prepare_for_std_eval(data=results)
        std_metrics = self.std_evaluator.evaluate(input_data)
        bleu=std_metrics['bleu']['damd']
        match=std_metrics['success']['inform']['total']
        success=std_metrics['success']['success']['total']
        score = 0.5 * (success + match) + bleu
        #logger = logging.getLogger()
        #logger.setLevel(logging.INFO)
        if cfg.mode=='test':
            logging.info(std_metrics)
        
        logging.info('[Std] validation %2.2f  %2.2f  %2.2f  %.2f  %.3f' % (match, success, bleu, score, joint_acc))
        json.dump(results, open(result_path, 'w'), indent=2)
        #self.reader.save_result('w', results, field,result_name='result.csv')

        eval_results = {}
        eval_results['bleu'] = std_metrics['bleu']['damd']
        eval_results['success'] = std_metrics['success']['success']['total']
        eval_results['match'] = std_metrics['success']['inform']['total']
        eval_results['score'] = score
        eval_results['joint_acc']=joint_acc
        cfg.batch_size=cfg.origin_batch_size
        return eval_results

    def generate_batch(self, model, contexts, max_len, eos_id, beam=1):
        # generate by batch
        # contexts: a list of ids
        # max_len: the max generated length
        # eos_id: the end id
        # return: a batch of ids with pre pad 
        batch_size=len(contexts)
        end_flag=np.zeros(batch_size)
        if beam>1:
            beam_box=[beam]*batch_size
            beam_result=[[] for _ in range(batch_size)]
            max_prob=[-float('inf')]*batch_size
        past_key_values=None
        inputs,attentions=self.reader.batch_align(contexts,left_len=max_len,return_attn=True)
        inputs=torch.tensor(inputs).to(model.device)
        attentions=torch.tensor(attentions).to(model.device)
        model.eval()
        with torch.no_grad():
            for i in range(max_len):
                if beam==1:
                    position_ids = attentions.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attentions == 0, 1)
                    if past_key_values is not None:
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                    if inputs.size(0)==0:
                        raise ValueError(contexts, inputs.cpu().list(), attentions)
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
                else:
                    if i==0:
                        position_ids = attentions.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attentions == 0, 1)
                        outputs=model(inputs,attention_mask=attentions,position_ids=position_ids,\
                                return_dict=True,use_cache=True,past_key_values=past_key_values)
                        past_key_values=[outputs.past_key_values]*beam
                        log_prob=F.log_softmax(outputs.logits[:, -1, :], -1) # B, V
                        beam_prob, beam_idx=torch.topk(log_prob, beam, -1) # B, beam
                        gen_tensor=beam_idx.unsqueeze(-1)# B, beam, 1
                        attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(model.device)),dim=1)
                        position_ids = attentions.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attentions == 0, 1)
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                        pv_beam_prob=beam_prob #B, beam
                        pv_beam_idx=beam_idx#B, beam
                    else:
                        for j in range(beam):
                            inputs=pv_beam_idx[:,j].unsqueeze(-1) # B, 1
                            outputs=model(inputs,attention_mask=attentions,position_ids=position_ids,\
                                return_dict=True,use_cache=True,past_key_values=past_key_values[j])
                            past_key_values[j]=outputs.past_key_values
                            log_prob=F.log_softmax(outputs.logits[:, -1, :], -1) # B, V
                            beam_prob, beam_idx=torch.topk(log_prob, beam, -1) # B, beam
                            if j==0:
                                prob_pool= beam_prob+pv_beam_prob[:, j].unsqueeze(-1).expand(-1, beam) # B, beam
                                id_pool=beam_idx
                            else:
                                prob_pool=torch.cat([prob_pool, beam_prob+pv_beam_prob[:, j].unsqueeze(-1).expand(-1, beam)],-1) # B, beam*beam
                                id_pool=torch.cat([id_pool, beam_idx], -1)# B, beam*beam
                        beam_prob, temp_id=torch.topk(prob_pool, beam, -1) #B, beam
                        beam_idx=torch.gather(id_pool, -1, temp_id)
                        temp_id=temp_id//beam
                        new_past_key_values=copy.deepcopy(past_key_values)
                        for b in range(batch_size):
                            gen_tensor[b, :, :]=gen_tensor[b, :, :].index_select(0, temp_id[b, :])
                            for t in range(beam):
                                for l in range(6):
                                    new_past_key_values[t][l][:, b, :,:,:]=past_key_values[temp_id[b, t]][l][:, b, :, :, :]
                        past_key_values=new_past_key_values
                        #past_key_values=[past_key_values[t] for t in temp_id.cpu().list()]
                        gen_tensor=torch.cat([gen_tensor, beam_idx.unsqueeze(-1)],-1) #B, beam, T
                        attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(model.device)),dim=1)
                        position_ids = attentions.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attentions == 0, 1)
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                        pv_beam_prob=beam_prob #B, beam
                        pv_beam_idx=beam_idx
                    for m in range(batch_size):
                        for n, gen in enumerate(gen_tensor.cpu().tolist()[m]):
                            if eos_id in gen:
                                beam_box[m]-=1
                                avg_prob=pv_beam_prob[m][n]/len(gen)
                                beam_result[m].append((gen, avg_prob))
                                pv_beam_prob[m][n]=-float('inf')
                    # we do not break during beam search
                    #if not any(beam_box):
                     #   break
            
        if beam==1:
            return gen_tensor.cpu().tolist()
        else:
            for i, tup in enumerate(beam_result):
                beam_list=sorted(tup, key=lambda item:item[1], reverse=True)
                beam_result[i]=[item[0] for item in beam_list[:beam]]
            return beam_result

    def generate_batch_e2e(self, batch):
        bs_max_len=75
        resp_max_len=80 if cfg.model_act else 60
        sos_b_id=self.sos_b_id
        eos_b_id=self.eos_b_id
        sos_r_id=self.sos_r_id
        eos_a_id=self.eos_a_id
        eos_r_id=self.eos_r_id
        
        batch_size=len(batch)
        contexts=[[] for i in range(batch_size)]
        bs_gen=[]
        db_gen=[]
        resp_gen=[]
        with torch.no_grad():
            for turn_num in range(len(batch[0])):
                past_key_values=None
                end_flag=np.zeros(len(batch))
                contexts=self.convert_eval_batch(batch,contexts,turn_num,bs_gen,\
                        prior=True,resp_gen=resp_gen)
                '''
                if self.global_output>0 and cfg.mode=='test':
                    logging.info(self.tokenizer.decode(contexts[0]))
                    self.global_output-=1
                '''
                inputs,attentions=self.reader.batch_align(contexts,left_len=bs_max_len,return_attn=True)
                inputs=torch.tensor(inputs).to(self.device1)
                attentions=torch.tensor(attentions).to(self.device1)
                if not cfg.use_true_curr_bspn:#generate
                    for i in range(bs_max_len):
                        position_ids = attentions.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attentions == 0, 1)
                        if past_key_values is not None:
                            position_ids=position_ids[:, -1].unsqueeze(-1)
                        outputs=self.PrioriModel(inputs,attention_mask=attentions,position_ids=position_ids,\
                                return_dict=True,use_cache=True,past_key_values=past_key_values)

                        past_key_values=outputs.past_key_values

                        preds=outputs.logits[:,-1,:].argmax(-1)#B
                        if i==0:
                            bs_tensor=preds.unsqueeze(1)
                        else:
                            bs_tensor=torch.cat([bs_tensor,preds.unsqueeze(1)],dim=1)
                        attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(self.device1)),dim=1)
                        inputs=preds.unsqueeze(1)
                        end_flag+=(preds.cpu().numpy()==eos_b_id).astype(float)
                        if sum(end_flag==0)==0:
                            break
                    bs_gen,db_gen=self.get_bspn(bs_tensor,return_db=True,data=batch,turn_num=turn_num)
                else:
                    for dial in batch:
                        bs_gen.append(dial[turn_num]['bspn'])
                        db_gen.append(dial[turn_num]['aspn'])
                past_key_values=None
                end_flag=np.zeros(len(batch))
                contexts=self.convert_eval_batch(batch,contexts,turn_num,bs_gen,\
                    prior=True,db_gen=db_gen)
                '''
                if self.global_output>0 and cfg.mode=='test':
                    logging.info(self.tokenizer.decode(contexts[0]))
                '''
                inputs,attentions=self.reader.batch_align(contexts,left_len=resp_max_len,return_attn=True)
                inputs=torch.tensor(inputs).to(self.device1)#B,T
                attentions=torch.tensor(attentions).to(self.device1)
                for i in range(resp_max_len):
                    position_ids = attentions.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attentions == 0, 1)
                    if past_key_values is not None:
                        position_ids=position_ids[:, -1].unsqueeze(-1)

                    outputs=self.PrioriModel(inputs,attention_mask=attentions,position_ids=position_ids,\
                            return_dict=True,use_cache=True,past_key_values=past_key_values)
                    past_key_values=outputs.past_key_values
                    preds=outputs.logits[:,-1,:].argmax(-1)#B
                    if i==0:
                        resp_tensor=preds.unsqueeze(1)
                    else:
                        resp_tensor=torch.cat([resp_tensor,preds.unsqueeze(1)],dim=1)
                    attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(self.device1)),dim=1)
                    inputs=preds.unsqueeze(1)
                    end_flag+=(preds.cpu().numpy()==eos_r_id).astype(float)
                    if sum(end_flag==0)==0:
                        break
                resp_gen=self.get_resp(resp_tensor)#if cfg.model_act then resp_gen contains act_gen and resp_gen
                for i in range(len(batch)):
                    batch[i][turn_num]['bspn_gen']=bs_gen[i]
                    #batch[i][turn_num]['db']=db_gen[i]
                    if cfg.model_act:
                        temp=resp_gen[i]
                        if eos_a_id in temp:
                            batch[i][turn_num]['aspn_gen']=temp[:temp.index(eos_a_id)+1]
                        else:
                            batch[i][turn_num]['aspn_gen']=temp[:-1]+[eos_a_id]
                        if sos_r_id in temp:
                            batch[i][turn_num]['resp_gen']=temp[temp.index(sos_r_id):]
                        else:
                            batch[i][turn_num]['resp_gen']=[sos_r_id]+temp[1:]
                        aspn_temp = batch[i][turn_num]['aspn'] if cfg.use_true_prev_aspn else batch[i][turn_num]['aspn_gen']
                        resp_temp = batch[i][turn_num]['resp'] if cfg.use_true_prev_resp else batch[i][turn_num]['resp_gen']
                        resp_gen[i] = aspn_temp+resp_temp
                    else:
                        batch[i][turn_num]['resp_gen']=resp_gen[i]
                        resp_gen[i] = batch[i][turn_num]['resp'] if cfg.use_true_prev_resp else batch[i][turn_num]['resp_gen']
        return batch
    
    def generate_batch_turn_level(self, batch):
        
        batch=self.reader.transpose_batch(batch)

        bs_max_len=75
        resp_max_len=80
        sos_b_id=self.sos_b_id
        eos_b_id=self.eos_b_id
        sos_r_id=self.sos_r_id
        eos_a_id=self.eos_a_id
        eos_r_id=self.eos_r_id

        batch_size=len(batch[0]['dial_id'])
        contexts=[[] for i in range(batch_size)]
        bs_gen=[]
        db_gen=[]
        resp_gen=[]
        pv_batch=None

        device=self.device1
        with torch.no_grad():
            for turn_num, turn_batch in enumerate(batch):
                # generate bspn
                past_key_values=None
                end_flag=np.zeros(batch_size)
                contexts=self.reader.convert_eval_batch_turn(turn_batch,pv_batch, mode='gen_bspn')
                '''
                if self.global_output>0 and cfg.mode=='test':
                    logging.info(self.tokenizer.decode(contexts[0]))
                    self.global_output-=1
                '''
                inputs,attentions=self.reader.batch_align(contexts,left_len=bs_max_len,return_attn=True)
                inputs=torch.tensor(inputs).to(device)
                attentions=torch.tensor(attentions).to(device)
                for i in range(bs_max_len):
                    position_ids = attentions.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attentions == 0, 1)
                    if past_key_values is not None:
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                    outputs=self.PrioriModel(inputs,attention_mask=attentions,position_ids=position_ids,\
                            return_dict=True,use_cache=True,past_key_values=past_key_values)

                    past_key_values=outputs.past_key_values

                    preds=outputs.logits[:,-1,:].argmax(-1)#B
                    if i==0:
                        bs_tensor=preds.unsqueeze(1)
                    else:
                        bs_tensor=torch.cat([bs_tensor,preds.unsqueeze(1)],dim=1)
                    attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(device)),dim=1)
                    inputs=preds.unsqueeze(1)
                    end_flag+=(preds.cpu().numpy()==eos_b_id).astype(float)
                    if sum(end_flag==0)==0:
                        break
                bs_gen,db_gen=self.get_bspn(bs_tensor,return_db=True,data=batch,turn_num=turn_num)
                # generate aspn and resp
                past_key_values=None
                end_flag=np.zeros(batch_size)
                contexts=self.reader.convert_eval_batch_turn(turn_batch,pv_batch, mode='gen_ar', 
                    bspn_gen=bs_gen,db_gen=db_gen)
                
                #if self.global_output>0 and cfg.mode=='test':
                 #   logging.info(self.tokenizer.decode(contexts[0]))
                inputs,attentions=self.reader.batch_align(contexts,left_len=resp_max_len,return_attn=True)
                inputs=torch.tensor(inputs).to(device)
                attentions=torch.tensor(attentions).to(device)
                for i in range(resp_max_len):
                    position_ids = attentions.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attentions == 0, 1)
                    if past_key_values is not None:
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                    outputs=self.PrioriModel(inputs,attention_mask=attentions,position_ids=position_ids,\
                            return_dict=True,use_cache=True,past_key_values=past_key_values)
                    past_key_values=outputs.past_key_values
                    preds=outputs.logits[:,-1,:].argmax(-1)#B
                    if i==0:
                        resp_tensor=preds.unsqueeze(1)
                    else:
                        resp_tensor=torch.cat([resp_tensor,preds.unsqueeze(1)],dim=1)
                    attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(device)),dim=1)
                    inputs=preds.unsqueeze(1)
                    end_flag+=(preds.cpu().numpy()==eos_r_id).astype(float)
                    if sum(end_flag==0)==0:
                        break
                resp_gen=self.get_resp(resp_tensor)
                aspn_gen=[]
                for i, temp in enumerate(resp_gen):
                    if eos_a_id in temp:
                        aspn=temp[:temp.index(eos_a_id)+1]
                    else:
                        aspn=temp[:-1]+[eos_a_id]
                    if sos_r_id in temp:
                        resp=temp[temp.index(sos_r_id):]
                    else:
                        resp=[sos_r_id]+temp[1:]
                    resp_gen[i]=resp
                    aspn_gen.append(aspn)
                pv_batch=self.reader.get_pv_batch(pv_batch, user=turn_batch['user'], resp=resp_gen, bspn=bs_gen)
                turn_batch['bspn_gen']=bs_gen
                turn_batch['aspn_gen']=aspn_gen
                turn_batch['resp_gen']=resp_gen
                turn_batch['db_gen']=db_gen
        return self.reader.inverse_transpose_batch(batch)

    def generate_batch_us(self, batch):
        batch=self.reader.transpose_batch(batch)
        max_len=75

        batch_size=len(batch[0]['dial_id'])
        contexts=[[] for i in range(batch_size)]
        bs_gen=[]
        db_gen=[]
        resp_gen=[]
        pv_batch=None
        device=self.model.device
        self.model.eval()
        with torch.no_grad():
            for turn_num, turn_batch in enumerate(batch):
                # we first generate aspn

                # we generate user act and user utterance together
                past_key_values=None
                end_flag=np.zeros(batch_size)
                contexts=self.reader.convert_eval_batch_turn_us(turn_batch, pv_batch)
                
                if self.global_output>0 and cfg.mode=='test':
                    logging.info(self.tokenizer.decode(contexts[0]))
                    self.global_output-=1
                
                inputs,attentions=self.reader.batch_align(contexts,left_len=max_len,return_attn=True)
                inputs=torch.tensor(inputs).to(device)
                attentions=torch.tensor(attentions).to(device)
                for i in range(max_len):
                    position_ids = attentions.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attentions == 0, 1)
                    if past_key_values is not None:
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                    outputs=self.model(inputs,attention_mask=attentions,position_ids=position_ids,\
                            return_dict=True,use_cache=True,past_key_values=past_key_values)

                    past_key_values=outputs.past_key_values
                    preds=outputs.logits[:,-1,:].argmax(-1)#B
                    if i==0:
                        u_tensor=preds.unsqueeze(1)
                    else:
                        u_tensor=torch.cat([u_tensor,preds.unsqueeze(1)],dim=1)
                    attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(device)),dim=1)
                    inputs=preds.unsqueeze(1)
                    end_flag+=(preds.cpu().numpy()==self.eos_u_id).astype(float)
                    if sum(end_flag==0)==0:
                        break
                u_gen=self.get_user(u_tensor)
                user_gen=[]
                usr_act_gen=[]
                for i, temp in enumerate(u_gen):
                    if self.eos_ua_id in temp:
                        usr_act=temp[:temp.index(self.eos_ua_id)+1]
                    else:
                        usr_act=temp[:-1]+[self.eos_ua_id]
                    if self.sos_u_id in temp:
                        user=temp[temp.index(self.sos_u_id):]
                    else:
                        user=[self.sos_u_id]+temp[1:]
                    user_gen.append(user)
                    usr_act_gen.append(usr_act)
                pv_batch=self.reader.get_pv_batch(pv_batch, resp=turn_batch['resp'], aspn=turn_batch['sys_act'], side='user')
                turn_batch['usr_act_gen']=usr_act_gen
                turn_batch['user_gen']=user_gen
        return self.reader.inverse_transpose_batch(batch)

    def validate_us(self, data='dev'):
        eval_data = self.reader.get_eval_data(data)
        if cfg.debugging:
            eval_data=eval_data[:100]
        result_path=os.path.join(cfg.eval_load_path, 'result.json')
        cfg.batch_size=cfg.eval_batch_size
        batches=self.reader.get_batches('test',data=eval_data)

        # valid_losses = []
        result_collection = []
        st=time.time()
        for batch in batches:
            try:
                if batch==[]:
                    continue
                batch=self.generate_batch_us(batch)
                result_collection+=self.reader.convert_batch_ids_to_tokens(batch)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    logging.info("WARNING: ran out of memory during validation and batch will be divided by half, batch size:{}, turn num:{}"\
                        .format(len(batch),len(batch[0])))
                    if hasattr(torch.cuda, 'empty_cache'):
                        with torch.cuda.device(self.device1):
                            torch.cuda.empty_cache()
                    #divide the batch in half if out of memory
                    batches.insert(0,batch[:len(batch)//2])
                    batches.insert(1,batch[len(batch)//2:])
                else:
                    logging.info(str(exception))
                    raise exception
        bleu, P, R, F1=self.evaluator.evaluate_us(result_collection)
        logging.info('BLEU:{:.2f}, Avg_Precious:{:.3f}, Avg_Recall:{:.3f}, Avg_F1:{:.3f}'.format(
            bleu, P, R, F1
        ))
        logging.info('Evaluation time:{:.2f} min'.format((time.time()-st)/60))
        json.dump(result_collection, open(result_path, 'w'), indent=2)
        return bleu, P, R, F1

def parse_arg_cfg(args):
    # add args to cfg
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
                if k == 'cuda_device':
                    v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    return

def fix_cfg():
    # if dataset is multiwoz2.1 then run this function
    assert cfg.dataset in [0,1]
    if cfg.dataset==0:
        cfg.data_path='./data/multi-woz-processed/'
        cfg.vocab_path_train = './data/multi-woz-processed/vocab'
        cfg.dev_list = 'data/multi-woz/valListFile.json'
        cfg.test_list='data/multi-woz/testListFile.json'
        cfg.domain_file_path = 'data/multi-woz-processed/domain_files.json'
        cfg.multi_acts_path = 'data/multi-woz-processed/multi_act_mapping_train.json'
    else:
        cfg.data_path='./data/multi-woz-2.1-processed/'
        cfg.vocab_path_train = './data/multi-woz-2.1-processed/vocab'

        cfg.dev_list = 'data/MultiWOZ_2.1/valListFile.txt'
        cfg.test_list='data/MultiWOZ_2.1/testListFile.txt'
        cfg.domain_file_path = 'data/multi-woz-2.1-processed/domain_files.json'
        cfg.multi_acts_path = 'data/multi-woz-2.1-processed/multi_act_mapping_train.json'
    if cfg.train_us:
        cfg.data_file='data_for_us.json'
    else:
        cfg.data_file='data_for_damd_fix.json'


def main():
    if not os.path.exists('./experiments'):
        os.mkdir('./experiments')

    if not os.path.exists('./experiments_21'):
        os.mkdir('./experiments_21')

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    cfg.mode = args.mode
    if 'test' in args.mode:
        parse_arg_cfg(args)
        cfg.eval_load_path=cfg.gpt_path
    else:  # train
        parse_arg_cfg(args)
        #print('exp_no:',cfg.exp_no)
        if cfg.exp_path in ['', 'to be generated']:
            experiments_path = './experiments_21' if cfg.dataset==1 else './experiments'
            if cfg.exp_no=='':
                if cfg.mode=='pretrain':
                    if cfg.posterior_train:
                        cfg.exp_no='pre_pos'
                    else:
                        cfg.exp_no='pre_'
                elif cfg.mode=='semi_ST':
                    cfg.exp_no='ST_'
                    if cfg.fix_ST:
                        cfg.exp_no=cfg.exp_no+'fix_'
                elif cfg.mode=='semi_VL':
                    cfg.exp_no='VL_'
                elif cfg.mode=='train':
                    cfg.exp_no='full'
                    if cfg.posterior_train:
                        cfg.exp_no = cfg.exp_no + '_pos'
                if cfg.mode!='train':
                    cfg.exp_no = cfg.exp_no + str(cfg.spv_proportion)
                if cfg.model_act:
                    cfg.exp_no = cfg.exp_no + '_act'
                if cfg.data_aug:
                    cfg.exp_no='full_aug_VL' if cfg.mode=='semi_VL' else 'full_aug_ST'
            print('exp_no:',cfg.exp_no)
            cfg.exp_path = os.path.join(experiments_path,'{}_{}_sd{}_lr{}_bs{}_ga{}'.format('-'.join(cfg.exp_domains),
                                                                          cfg.exp_no, cfg.seed, cfg.lr, cfg.batch_size,
                                                                          cfg.gradient_accumulation_steps))
            if 'test' not in cfg.mode:
                print('save path:', cfg.exp_path)
            if cfg.save_log:
                if not os.path.exists(cfg.exp_path):
                    os.mkdir(cfg.exp_path)

            # to gpt later
            cfg.model_path = os.path.join(cfg.exp_path, 'model.pkl')
            cfg.result_path = os.path.join(cfg.exp_path, 'result.csv')
            cfg.vocab_path_eval = os.path.join(cfg.exp_path, 'vocab')
            cfg.eval_load_path = cfg.exp_path

    fix_cfg()
    cfg._init_logging_handler(args.mode)
    logging.info('Model path:{}'.format(cfg.eval_load_path))
    device=cfg.cuda_device
    cfg.divided_path=os.path.join(cfg.data_path,'divided_data{}.json'.format(cfg.spv_proportion))

    # fix random seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # initialize model
    m = Modal(device)

    if args.mode =='pretrain' or args.mode=='train':
        if cfg.train_modular:
            m.train_modular(modular=cfg.modular)
        elif cfg.train_sys:
            m.pretrain_sys_model()
        elif cfg.turn_level:
            m.pretrain_turn_level()
        else:
            m.pretrain(posterior=cfg.posterior_train)
    else:  # test
        if cfg.train_modular:
            if cfg.combine_eval:
                m.combine_modules()
            else:
                m.eval_modular(cfg.modular)
        elif cfg.train_us:
            m.validate_us('test')
        else:
            logging.info("Generate setting: \n\t use true_prev_bspn={} \n\t use true_prev_aspn={} \n\t use true_db_pointer={} \n\t use true_prev_resp={} \n\t use true_curr_bspn={} \n\t use true_curr_aspn={} \n\t use_all_previous_context={}".format(
                                cfg.use_true_prev_bspn, cfg.use_true_prev_aspn, cfg.use_true_db_pointer, cfg.use_true_prev_resp,
                                cfg.use_true_curr_bspn, cfg.use_true_curr_aspn, cfg.use_all_previous_context
                            ))
            m.validate_fast('test')


if __name__ == "__main__":
    main()
