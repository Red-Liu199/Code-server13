from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from eval import MultiWozEvaluator
#from damd_net import DAMD, cuda_, get_one_hot_input
from reader import MultiWozReader
import utils
from torch.optim import Adam
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        if cfg.mode=='semi_VL':
            logging.info('PrioriModel sets on GPU{}, PosteriorModel sets on GPU{}'.format(self.device1,self.device2))
            tokenizer_path=cfg.PrioriModel_path
        else:
            tokenizer_path=cfg.gpt_path
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        # cfg.tokenizer = tokenizer

        # initialize multiwoz reader
        if cfg.data_aug and cfg.mode=='semi_ST':#self training cannot generate accurate database result 
            cfg.dbs={
            'attraction': '../data/schema-guided/attraction_db.json',
            'hospital': 'db/hospital_db_processed.json',
            'hotel': '../data/schema-guided/hotel_db.json',
            'police': 'db/police_db_processed.json',
            'restaurant': '../data/schema-guided/restaurant_db.json',
            'taxi': '../data/schema-guided/taxi_db.json',
            'train': '../data/schema-guided/train_db.json',
            }
        logging.info('hotel database path:{}'.format(cfg.dbs['hotel']))
        self.reader = MultiWozReader(self.tokenizer)
        self.sos_b_id=self.tokenizer.convert_tokens_to_ids('<sos_b>')
        self.sos_a_id=self.tokenizer.convert_tokens_to_ids('<sos_a>')
        self.sos_r_id=self.tokenizer.convert_tokens_to_ids('<sos_r>')
        self.eos_b_id=self.tokenizer.convert_tokens_to_ids('<eos_b>')
        self.eos_a_id=self.tokenizer.convert_tokens_to_ids('<eos_a>')
        self.eos_r_id=self.tokenizer.convert_tokens_to_ids('<eos_r>')
        self.sos_db_id=self.tokenizer.convert_tokens_to_ids('<sos_db>')
        self.eos_db_id=self.tokenizer.convert_tokens_to_ids('<eos_db>')
        logging.info([self.sos_b_id, self.sos_a_id, self.sos_r_id, self.eos_b_id, self.eos_a_id,self.eos_r_id])

        # create model: gpt2
        single_mode=['pretrain','train','semi_ST','test_pos']
        if cfg.mode in single_mode:
            self.model=GPT2LMHeadModel.from_pretrained(cfg.gpt_path)
            self.model.resize_token_embeddings(len(self.tokenizer))
            if cfg.gradient_checkpoint:
                self.model.config.gradient_checkpointing=True
            
            self.model.to(self.device1)
            self.PrioriModel=self.model
            if cfg.posterior_train:
                logging.info("Posterior model loaded from {}".format(cfg.gpt_path))
            else:
                logging.info("Prior model loaded from {}".format(cfg.gpt_path))
        
        elif cfg.mode=='test' or cfg.mode=='test_all':
            self.PrioriModel=GPT2LMHeadModel.from_pretrained(cfg.gpt_path)
            self.model=self.PrioriModel
            if cfg.gradient_checkpoint:
                self.PrioriModel.config.gradient_checkpointing=True
            self.PosteriorModel=None
            self.PrioriModel.to(self.device1)
        
        elif cfg.mode=='semi_VL':#semi-VL
            self.PrioriModel=GPT2LMHeadModel.from_pretrained(cfg.PrioriModel_path)
            self.PosteriorModel=GPT2LMHeadModel.from_pretrained(cfg.PosteriorModel_path)
            logging.info("model loaded from {} and {}".format(cfg.PrioriModel_path,cfg.PosteriorModel_path))
            self.PrioriModel.resize_token_embeddings(len(self.tokenizer))
            self.PosteriorModel.resize_token_embeddings(len(self.tokenizer))
            if cfg.gradient_checkpoint:
                self.PrioriModel.config.gradient_checkpointing=True
                self.PosteriorModel.config.gradient_checkpointing=True
            self.PrioriModel.to(self.device1)
            self.PosteriorModel.to(self.device2)

        self.vocab_size=len(self.tokenizer)
        #
        self.evaluator = MultiWozEvaluator(self.reader)
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

        self.nll_loss=nn.NLLLoss(ignore_index=cfg.pad_id)
        self.eps=1e-45
        if 'test' not in cfg.mode:
            json.dump(cfg.__dict__,open(os.path.join(cfg.exp_path,'cfg_all.json'),'w'),indent=2)
        self.global_output=4
    
    def pretrain(self, posterior=False):
        #logging.info(cfg.mode)
        if cfg.mode=='train':
            num_dials=len(self.reader.train)
            all_batches = self.reader.get_batches('train')
        else:
            #divide the datasets
            if not os.path.exists(cfg.divided_path):
                train_data=self.reader.train
                temp_path=os.path.join(cfg.data_path,'divided_data{}.json'.format(cfg.spv_proportion-5))
                #logging.info(temp_path)
                if os.path.exists(temp_path):
                    encoded_data = json.loads(open(temp_path, 'r', encoding='utf-8').read())
                    add_len=int(0.05*len(train_data))
                    self.pre_data=encoded_data['pre_data']+encoded_data['post_data'][:add_len]
                    self.post_data=encoded_data['post_data'][add_len:]
                    encoded_data={'pre_data':self.pre_data,'post_data':self.post_data}
                    logging.info('Divide data from %s, saved in %s'%(temp_path, cfg.divided_path))
                    json.dump(encoded_data, open(cfg.divided_path, 'w'), indent=2)
                else:
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
                    loss=self.calculate_loss_and_accuracy(outputs,labels['contexts_tensor'])
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

    def pretrain_turn_level(self, posterior=False):
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
        warmup_epochs=cfg.warmup_steps*cfg.gradient_accumulation_steps*cfg.batch_size//num_dials \
            if cfg.warmup_steps>=0 else int(cfg.epoch_num*cfg.warmup_ratio)

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
                    inputs, labels = self.reader.convert_batch_turn(turn_batch, pv_batch, first_turn)
                    pv_batch = self.reader.get_pv_batch(pv_batch, turn_batch['user'],
                        turn_batch['resp'], turn_batch['bspn'])
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

                            logs = {}  # for tb writer
                            # logging: loss, lr... after certain amount of steps
                            

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
    

    def semi_ST(self):
        logging.info('------Running self training------')
        data=json.loads(open(cfg.divided_path, 'r', encoding='utf-8').read())
        data_lab=data['pre_data']
        data_unl=data['post_data']
            
        logging.info('Labeled dials:{}, unlabeled dials:{}'.format(len(data_lab),len(data_unl)))
        num_dials=len(data_lab)+len(data_unl)
        
        cfg.batch_size=cfg.batch_size*cfg.gradient_accumulation_steps
        batches_lab=self.reader.get_batches('train',data=data_lab)
        batches_unl=self.reader.get_batches('train',data=data_unl)
        all_batches=[]
        data_repeate=3 if cfg.spv_proportion==10 else 1
        for _ in range(data_repeate-1):
            num_dials+=len(data_lab)

        for _ in range(data_repeate):
            for batch in batches_lab:
                all_batches.append({'batch':batch,'supervised':True})
        
        for batch in batches_unl:
            all_batches.append({'batch':batch,'supervised':False})

        optimizer, scheduler = self.get_sep_optimizers(num_dials,self.model)

        # log info
        logging.info("  Num Dialogs = %d", num_dials)
        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)
        logging.info('  Num Batches = %d', len(all_batches))


        log_inputs = 0
        global_step = 0
        min_loss = 1000
        min_eval_loss=1000
        max_score=0
        early_stop_count=cfg.early_stop_count
        if cfg.use_scheduler:
            warmup_epochs=cfg.warmup_steps*cfg.batch_size//num_dials if cfg.warmup_steps>=0 else int(cfg.epoch_num*cfg.warmup_ratio)
            logging.info('Warmup epochs:{}'.format(warmup_epochs))
        #logging.info('the initial evaluation result:')
        #eval_result=self.validate_fast(data='dev')
        weight_decay_count=cfg.weight_decay_count
        lr=cfg.lr
        for epoch in tqdm(range(cfg.epoch_num)):
            epoch_step = 0
            total_loss=0
            logging_loss=0
            btm = time.time()
            oom_time = 0
            #shuffle batch instead of data
            random.shuffle(all_batches)
            for batch_idx, dial_batch_dict in enumerate(all_batches):
                self.model.zero_grad()
                turn_nums=[len(dial) for dial in dial_batch_dict['batch']]
                consists=[turn_num==turn_nums[0] for turn_num in turn_nums]
                assert all(consists)
                if dial_batch_dict['supervised']==False:
                    dial_batch_large=self.gen_batch_bspn(dial_batch_dict['batch'])
                else:
                    dial_batch_large=dial_batch_dict['batch']
                dial_batch=[]
                for i, dial in enumerate(dial_batch_large):
                    dial_batch.append(dial)
                    if len(dial_batch)==cfg.origin_batch_size or i==len(dial_batch_large)-1:
                        if dial_batch_dict['supervised']:
                            resp_only=False
                        else:
                            resp_only=cfg.ST_resp_only
                        inputs, labels,bspn_labels = self.reader.convert_batch_session(dial_batch,\
                            posterior_train=False,only_resp_label=resp_only,bspn_label=True)
                        try:
                            self.model.train()
                            if log_inputs > 0 and cfg.example_log:  # log inputs for the very first two turns
                                logging.info('examples')
                                logging.info(self.tokenizer.decode(inputs['contexts'][0]))
                                log_inputs-=1
                            inputs = self.add_torch_input(inputs)#B,T
                            labels=self.add_torch_input(labels)#B,T
                            bspn_labels=self.add_torch_input(bspn_labels)
                            bspn_labels=bspn_labels['contexts_tensor']
                            outputs = self.model(inputs['contexts_tensor'])
                            if cfg.fix_ST and not dial_batch_dict['supervised']:
                                ST_inputs=self.get_ST_input(inputs['contexts_tensor'],outputs[0],bspn_labels,bspn_labels)
                                embeds=ST_inputs.matmul(self.model.get_input_embeddings().weight)
                                outputs=self.model(inputs_embeds=embeds)
                            loss=self.calculate_loss_and_accuracy(outputs,labels['contexts_tensor'])
                            if cfg.loss_reg:
                                loss=loss/cfg.gradient_accumulation_steps
                            loss.backward()
                            total_loss+=loss.item()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                            epoch_step += 1
                        except RuntimeError as exception:
                            if "out of memory" in str(exception):
                                max_length = max(inputs['lengths'])
                                oom_time += 1
                                logging.info("WARNING: ran out of memory,times: {}, batch size: {}, max_len: {}".format(
                                    oom_time, len(dial_batch), max_length))
                            if hasattr(torch.cuda, 'empty_cache'):
                                with torch.cuda.device(self.model.device):
                                    torch.cuda.empty_cache()
                            else:
                                logging.info(str(exception))
                                raise exception
                        dial_batch=[]
                optimizer.step()
                if cfg.use_scheduler:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                loss_scalar = (total_loss - logging_loss) / cfg.gradient_accumulation_steps    
                logging_loss = total_loss
                if self.tb_writer:
                    self.tb_writer.add_scalar('lr', optimizer.param_groups[0]["lr"],global_step)
                    self.tb_writer.add_scalar('loss', loss_scalar, global_step)

            logging.info('Epoch:{}, Train epoch time: {:.2f} min, loss: {}'.format( 
                epoch, (time.time()-btm)/60, total_loss/epoch_step))
            # save model after every epoch
            if cfg.evaluate_during_training:
                eval_loss=self.eval(model=self.model)
                logging.info('Model evaluation loss:{}'.format(eval_loss))
                eval_result=self.validate_fast(data='dev')
                if self.tb_writer:
                    self.tb_writer.add_scalar('loss_eval',eval_loss,epoch)
                    self.tb_writer.add_scalar('joint_goal',eval_result['joint_acc'],epoch)
                    self.tb_writer.add_scalar('match',eval_result['match'],epoch)
                    self.tb_writer.add_scalar('success',eval_result['success'],epoch)
                    self.tb_writer.add_scalar('bleu',eval_result['bleu'],epoch)
                    self.tb_writer.add_scalar('combined_score',eval_result['score'],epoch)
                if eval_result['score']>max_score:
                    early_stop_count=cfg.early_stop_count
                    max_score=eval_result['score']
                    self.save_model(path='best_score_model',model=self.model)
                else:
                    weight_decay_count-=1
                    if weight_decay_count==0 and not cfg.use_scheduler:
                        lr=lr*cfg.lr_decay
                        for group in optimizer.param_groups:
                            group['lr'] = lr
                        logging.info("learning rate decay to {}".format(lr))
                        weight_decay_count = cfg.weight_decay_count
                    if epoch>=warmup_epochs:
                        early_stop_count-=1
                        logging.info('early stop count:%d'%early_stop_count)
                if lr<1e-9 and not cfg.use_scheduler:
                    logging.info('learning rate too small, break')
                    break
                if early_stop_count==0 and cfg.early_stop:
                    logging.info('early stopped')
                    break
            
            else:
                if total_loss/epoch_step<min_loss:
                    min_loss=total_loss/epoch_step
                    self.save_model(posterior=False,model=self.model)

    def semi_VL(self):
        
        logging.info('Reading encoded data from %s'%cfg.divided_path)
        data = json.loads(open(cfg.divided_path,'r', encoding='utf-8').read())
        data_lab=data['pre_data']
        data_unl=data['post_data']
        logging.info('Labeled dials:{}, unlabeled dials:{}'.format(len(data_lab),len(data_unl)))
        num_dials=len(data_lab)+len(data_unl)

        cfg.batch_size=cfg.batch_size*cfg.gradient_accumulation_steps
        batches_lab=self.reader.get_batches('train',data=data_lab)
        all_batches=[]
        data_repeate=3 if cfg.spv_proportion==10 else 1
        for _ in range(data_repeate-1):
            num_dials+=len(data_lab)

        for _ in range(data_repeate):
            for batch in batches_lab:
                all_batches.append({'batch':batch,'supervised':True})
        if cfg.delex_as_damd:
            batches_unl=self.reader.get_batches('train',data=data_unl)
            for batch in batches_unl:
                all_batches.append({'batch':batch,'supervised':False,'dataset':'all'})
        else:
            batches_unl1=self.reader.get_batches('train',data=data1)
            batches_unl2=self.reader.get_batches('train',data=data2)
            for batch in batches_unl1:
                all_batches.append({'batch':batch,'supervised':False, 'dataset':'TM'})
            for batch in batches_unl2:
                all_batches.append({'batch':batch,'supervised':False, 'dataset':'SGD'})
        optimizer1, scheduler1 = self.get_sep_optimizers(num_dials,self.PrioriModel)
        optimizer2, scheduler2 = self.get_sep_optimizers(num_dials,self.PosteriorModel)


        # log info
        logging.info("***** Running training *****")
        logging.info("  Num Dialogs = %d", num_dials)
        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)
        logging.info("  Gradient Accumulation steps = %d",
                     cfg.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d",
                     num_dials*cfg.epoch_num // (cfg.gradient_accumulation_steps*cfg.batch_size))


        log_inputs = 2
        global_step = 0
        max_bleu=0
        max_score=0
        min_loss=1000
        epoch_num=cfg.epoch_num
        early_stop_count=cfg.early_stop_count
        warmup_epochs=cfg.warmup_steps*cfg.batch_size//num_dials if cfg.warmup_steps>=0 else int(cfg.epoch_num*cfg.warmup_ratio)
        if cfg.use_scheduler:
            logging.info('warmup epochs:{}'.format(warmup_epochs))
        #eval_loss=self.eval(model=self.PrioriModel)
        #logging.info('the initial evaluation result:')
        #eval_result=self.validate_fast(data='dev')
        weight_decay_count=cfg.weight_decay_count
        lr=cfg.lr
        for epoch in tqdm(range(epoch_num)):
            epoch_step = 0
            epoch_step_uns=0
            epoch_step_sup=0
            tr_loss = 0.0
            loss1=0
            loss2=0
            loss_uns=0
            loss_sup=0
            logging_loss = 0.0
            btm = time.time()

            random.shuffle(all_batches)
            for batch_idx, dial_batch_dict in enumerate(all_batches):
                self.PrioriModel.zero_grad()
                self.PosteriorModel.zero_grad()
                turn_nums=[len(dial) for dial in dial_batch_dict['batch']]
                assert all([turn_num==turn_nums[0] for turn_num in turn_nums])
                if dial_batch_dict['supervised']==False:
                    #cfg.gen_db=True if dial_batch_dict['dataset']=='TM' else False
                    cfg.gen_db=True
                    dial_batch_large=self.gen_batch_bspn(dial_batch_dict['batch'],model_name='PosteriorModel')
                    cfg.gen_db=False
                    dial_batch=[]
                    for i, dial in enumerate(dial_batch_large):
                        dial_batch.append(dial)
                        if len(dial_batch)==cfg.origin_batch_size or i==len(dial_batch_large)-1:
                            try:
                                only_resp=True if cfg.VL_with_kl else False
                                inputs_prior, labels_prior, bspn_labels_pri = \
                                    self.reader.convert_batch_session(dial_batch,only_resp_label=only_resp,bspn_label=True)
                                inputs_posterior, labels_posterior = \
                                    self.reader.convert_batch_session(dial_batch,posterior_train=True)

                                self.PrioriModel.train()
                                self.PosteriorModel.train()
                                if log_inputs > 0 and cfg.example_log:  # log inputs for the very first two turns
                                    logging.info('Prior examples')
                                    logging.info(self.tokenizer.decode(inputs_prior['contexts'][0]))
                                    logging.info("Posterior examples")
                                    logging.info(self.tokenizer.decode(inputs_posterior['contexts'][0]))
                                    log_inputs -= 1

                                # to tensor
                                inputs_prior = self.add_torch_input(inputs_prior)#B,T
                                inputs_posterior = self.add_torch_input(inputs_posterior,posterior=True)
                                labels_prior=self.add_torch_input(labels_prior)#B,T
                                labels_posterior=self.add_torch_input(labels_posterior,posterior=True)
                                bspn_labels_pri=self.add_torch_input(bspn_labels_pri)
                                # loss
                                outputs_prior=self.PrioriModel(inputs_prior['contexts_tensor'])
                                outputs_posterior=self.PosteriorModel(inputs_posterior['contexts_tensor'])#B,T,V
                                logits_pri=outputs_prior[0]
                                logits_post=outputs_posterior[0]
                                #straight through trick
                                ST_inputs_prior=self.get_ST_input(inputs_prior['contexts_tensor'],\
                                        logits_post,bspn_labels_pri['contexts_tensor'],labels_posterior['contexts_tensor'])
                                if cfg.VL_with_kl:
                                    loss_kl=self.get_kl_loss(logits_pri,logits_post,\
                                        bspn_labels_pri['contexts_tensor'],labels_posterior['contexts_tensor'])
                                else:
                                    loss_kl=0
                                embed_prior=ST_inputs_prior.matmul(self.PrioriModel.get_input_embeddings().weight)#multiple the input embedding
                                outputs1=self.PrioriModel(inputs_embeds=embed_prior)
                                loss_ce=self.calculate_loss_and_accuracy(outputs1,labels_prior['contexts_tensor'])
                            
                                loss=loss_ce+cfg.kl_loss_weight*loss_kl
                                if cfg.loss_reg:
                                    loss=loss/cfg.gradient_accumulation_steps
                                loss.backward()
                                tr_loss += loss.item()
                                loss_uns+=loss.item()
                                loss1+=loss_ce.item()
                                if loss_kl!=0:
                                    loss2+=loss_kl.item()
                                torch.nn.utils.clip_grad_norm_(self.PrioriModel.parameters(), 5.0)
                                torch.nn.utils.clip_grad_norm_(self.PosteriorModel.parameters(), 5.0)
                                epoch_step += 1
                                dial_batch=[]
                                epoch_step_uns+=1
                            except RuntimeError as exception:
                                if "out of memory" in str(exception):
                                    logging.info("WARNING: ran out of memory during unsupervised train, batch idx:{}, batch size:{}, turn num:{}"\
                                        .format(batch_idx,len(dial_batch),len(dial_batch[0])))
                                    if hasattr(torch.cuda, 'empty_cache'):
                                        with torch.cuda.device(self.device1):
                                            torch.cuda.empty_cache(self.device1)
                                        with torch.cuda.device(self.device2):
                                            torch.cuda.empty_cache(self.device2)
                                else:
                                    logging.info(str(exception))
                                    raise exception
                    
                    optimizer1.step()
                    optimizer1.zero_grad()
                    optimizer2.step()
                    optimizer2.zero_grad()
                    if cfg.use_scheduler:
                        scheduler1.step()
                        scheduler2.step()
                    global_step+=1
                    loss_scalar = (tr_loss - logging_loss) / cfg.gradient_accumulation_steps
                    logging_loss = tr_loss
                    if self.tb_writer:
                        self.tb_writer.add_scalar('lr1', optimizer1.param_groups[0]["lr"],global_step)
                        self.tb_writer.add_scalar('lr2', optimizer2.param_groups[0]["lr"],global_step)
                        self.tb_writer.add_scalar('loss', loss_scalar, global_step)
                else:
                    dial_batch_large=dial_batch_dict['batch']
                    dial_batch=[]
                    for i, dial in enumerate(dial_batch_large):
                        dial_batch.append(dial)
                        if len(dial_batch)==cfg.origin_batch_size or i==len(dial_batch_large)-1:
                            try:
                                self.PrioriModel.train()
                                self.PosteriorModel.train()
                                inputs_prior, labels_prior = self.reader.convert_batch_session(dial_batch,posterior_train=False)
                                inputs_posterior, labels_posterior = self.reader.convert_batch_session(dial_batch,posterior_train=True)
                                inputs_prior = self.add_torch_input(inputs_prior)#B,T
                                labels_prior=self.add_torch_input(labels_prior)#B,T
                                inputs_posterior=self.add_torch_input(inputs_posterior,posterior=True)
                                labels_posterior=self.add_torch_input(labels_posterior,posterior=True)

                                outputs1 = self.PrioriModel(inputs_prior['contexts_tensor'])
                                loss_pri=self.calculate_loss_and_accuracy(outputs1,labels_prior['contexts_tensor'])
                                outputs2=self.PosteriorModel(inputs_posterior['contexts_tensor'])
                                loss_pos=self.calculate_loss_and_accuracy(outputs2,labels_posterior['contexts_tensor'])

                                if cfg.loss_reg:
                                    loss_pri=loss_pri/cfg.gradient_accumulation_steps
                                    loss_pos=loss_pos/cfg.gradient_accumulation_steps
                                loss_pri.backward()
                                loss_pos.backward()
                                tr_loss+=loss_pri.item()+loss_pos.item()
                                loss_sup+=loss_pri.item()+loss_pos.item()
                                
                                torch.nn.utils.clip_grad_norm_(self.PrioriModel.parameters(), 5.0)
                                torch.nn.utils.clip_grad_norm_(self.PosteriorModel.parameters(), 5.0)
                                dial_batch=[]
                                epoch_step+=1
                                epoch_step_sup += 1
                            except RuntimeError as exception:
                                if "out of memory" in str(exception):
                                    logging.info("WARNING: ran out of memory during supervised train, batch idx:{}, batch size:{}, turn num:{}"\
                                        .format(batch_idx,len(dial_batch),len(dial_batch[0])))
                                    if hasattr(torch.cuda, 'empty_cache'):
                                        with torch.cuda.device(self.device1):
                                            torch.cuda.empty_cache(self.device1)
                                        with torch.cuda.device(self.device2):
                                            torch.cuda.empty_cache(self.device2)
                                else:
                                    logging.info(str(exception))
                                    raise exception
                            
                    optimizer1.step()
                    optimizer1.zero_grad()
                    optimizer2.step()
                    optimizer2.zero_grad()
                    if cfg.use_scheduler:
                        scheduler1.step()
                        scheduler2.step()
                    global_step+=1
                    loss_scalar = (tr_loss - logging_loss) / cfg.gradient_accumulation_steps
                    logging_loss = tr_loss
                    if self.tb_writer:
                        self.tb_writer.add_scalar('loss', loss_scalar, global_step)
 
            if epoch==0:
                logging.info('sup steps:{}, uns steps:{}'.format(epoch_step_sup,epoch_step_uns))

            logging.info('Epoch: {}, Train epoch time: {} min, loss:{}, loss_sup:{}, loss_uns:{}'.format(
                epoch, (time.time()-btm)/60, tr_loss/epoch_step, loss_sup/epoch_step_sup,loss_uns/epoch_step_uns))
            if self.tb_writer:
                self.tb_writer.add_scalar('loss_sup',loss_sup/epoch_step_sup,epoch)
                self.tb_writer.add_scalar('loss_uns',loss_uns/epoch_step_uns,epoch)
                self.tb_writer.add_scalar('loss_ce',loss1/epoch_step_uns,epoch)
                self.tb_writer.add_scalar('loss_kl',loss2/epoch_step_uns,epoch)
            
            if cfg.evaluate_during_training:
                eval_loss=self.eval(model=self.PrioriModel)
                logging.info('Prior model evaluation loss:{}'.format(eval_loss))
                eval_result=self.validate_fast(data='dev')
                if self.tb_writer:
                    self.tb_writer.add_scalar('loss_eval',eval_loss,epoch)
                    self.tb_writer.add_scalar('joint_goal',eval_result['joint_acc'],epoch)
                    self.tb_writer.add_scalar('match',eval_result['match'],epoch)
                    self.tb_writer.add_scalar('success',eval_result['success'],epoch)
                    self.tb_writer.add_scalar('bleu',eval_result['bleu'],epoch)
                    self.tb_writer.add_scalar('combined_score',eval_result['score'],epoch)
                
                if eval_result['score']>max_score:
                    max_score=eval_result['score']
                    self.save_model(path='best_score_model')
                    early_stop_count=cfg.early_stop_count
                else:
                    weight_decay_count-=1
                    if weight_decay_count==0 and not cfg.use_scheduler:
                        lr=lr*cfg.lr_decay
                        for group in optimizer1.param_groups:
                            group['lr'] = lr
                        for group in optimizer2.param_groups:
                            group['lr'] = lr
                        logging.info("learning rate decay to {}".format(lr))
                        weight_decay_count = cfg.weight_decay_count
                    if epoch>=warmup_epochs:
                        early_stop_count-=1
                        logging.info('early stop count:%d'%early_stop_count)
                if lr<1e-9 and not cfg.use_scheduler:
                    logging.info('learning rate too small, break')
                    break
                if early_stop_count==0 and cfg.early_stop:
                    logging.info('early stopped')
                    break
            else:
                if loss1/epoch_step<min_loss1:
                    min_loss1=loss1/epoch_step
                    self.save_model()
                if loss2/epoch_step<min_loss2:
                    min_loss2=loss2/epoch_step
                    self.save_model(posterior=True)
    

    def get_ST_input(self,inputs,logits,labels1,labels2):
        #inputs:B,T1
        #logits:B,T1,V or B,T2,V
        #labels1:B,T1
        #labels2:B,T1 or B,T2
        onehot=F.one_hot(inputs,self.vocab_size).float()
        for dial_idx in range(logits.size(0)):
            label_pri=labels1[dial_idx,:].ne(cfg.pad_id).long().cpu().tolist()#0 for pad token and 1 for hidden states tokens
            label_post=labels2[dial_idx,:].ne(cfg.pad_id).long().cpu().tolist()
            label_pri.reverse()#Traverse from back to front
            label_post.reverse()
            loc1=0
            loc2=0
            loc3=0
            loc4=0
            while(1):
                if 1 not in label_pri:
                    break
                loc1=label_pri.index(1)+loc2
                label_pri=label_pri[loc1-loc2:]
                if 0 not in label_pri:
                    break
                loc2=label_pri.index(0)+loc1
                if 1 not in label_post:
                    break
                loc3=label_post.index(1)+loc4
                label_post=label_post[loc3-loc4:]
                if 0 not in label_post:
                    break
                loc4=label_post.index(0)+loc3
                if (loc4-loc3)!=(loc2-loc1):
                    print('location:',loc1,loc2,loc3,loc4)
                assert loc4-loc3==loc2-loc1
                probs=F.softmax(logits[dial_idx,-loc4:-loc3-1,:])
                if loc1==0:
                    onehot[dial_idx,-loc2+1:,:]+=(probs-probs.detach()).to(onehot.device)
                else:
                    onehot[dial_idx,-loc2+1:-loc1,:]+=(probs-probs.detach()).to(onehot.device)
                label_pri=label_pri[loc2-loc1:]
                label_post=label_post[loc4-loc3:]
        return onehot

    def kl_loss(self, p_proba, q_proba): # [B, T, V] or [T,V]
        dim=p_proba.dim()
        loss = q_proba * (torch.log(q_proba+self.eps) - torch.log(p_proba+self.eps))
        loss = torch.sum(loss, dim=-1)   # sum over vocabulary
        loss = torch.sum(loss, dim=-1)   # sum over sequence
        if dim==2:
            return loss
        else:
            return loss.mean()

    def get_kl_loss(self,logits_pri,logits_post,labels_pri,labels_post):
        # logits_pri:B,T1,V
        # logits_post:B,T2,V
        # labels_pri:B,T1. bspn's label in prior sequence
        # labels_post:B,T2. bspn's label in posterior sequence
        # what labels do is to find the logits corresponding to bspn
        loss=0
        count=0
        for dial_idx in range(logits_pri.size(0)):
            label_pri=labels_pri[dial_idx,:].ne(cfg.pad_id).long().cpu().tolist()#pad_id处为0，bspn为1
            label_post=labels_post[dial_idx,:].ne(cfg.pad_id).long().cpu().tolist()
            label_pri.reverse()#从后往前遍历
            label_post.reverse()
            turn_count=0
            loc1=0
            loc2=0
            loc3=0
            loc4=0
            while(1):
                if 1 not in label_pri:
                    break
                loc1=label_pri.index(1)+loc2
                label_pri=label_pri[loc1-loc2:]
                if 0 not in label_pri:
                    break
                loc2=label_pri.index(0)+loc1
                if 1 not in label_post:
                    break
                loc3=label_post.index(1)+loc4
                label_post=label_post[loc3-loc4:]
                if 0 not in label_post:
                    break
                loc4=label_post.index(0)+loc3
                bspn_len=min(loc2-loc1,loc4-loc3)
                probs_pri=F.softmax(logits_pri[dial_idx,-(loc1+bspn_len):-loc1-1,:],dim=-1)
                probs_post=F.softmax(logits_post[dial_idx,-(loc3+bspn_len):-loc3-1,:],dim=-1)
                loss+=self.kl_loss(probs_pri,probs_post.to(probs_pri.device))
                count+=bspn_len
                turn_count+=1
                label_pri=label_pri[loc2-loc1:]
                label_post=label_post[loc4-loc3:]
        
        return loss/count


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

    def get_max_len(self,batch):
        max_len=0
        for dial in batch:
            dial_len=0
            for turn in dial:
                dial_len+=len(turn['user'])+len(turn['resp'])
            if dial_len>max_len:
                max_len=dial_len
        return max_len
    

    def convert_eval_batch(self, data, contexts, turn_num,bs_gen,prior=False,db_gen=None,resp_gen=None,aspn_gen=None, gen_db=False):
        
        if gen_db:#在使用后验网络生成数据库结果时使用
            new_contexts=[]
            for id, context in enumerate(contexts):
                new_contexts.append(context[:-1] + bs_gen[id] + [self.sos_db_id])
            return new_contexts
        else:
            for id,context in enumerate(contexts):
                if turn_num==0:
                    if prior:
                        if db_gen is None:#还没有生成bs_gen以及对应的db_gen
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
                    #context中已经含有sos_b了
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

    def gen_batch_bspn(self,original_batch,model_name=None,validate=False):
        if model_name=='PosteriorModel':
            self.model=self.PosteriorModel
        elif model_name=='PrioriModel':
            self.model=self.PrioriModel

        if cfg.mode=='test_pos' or validate or model_name=='PosteriorModel':
            prior=False
        else:
            prior=True
        self.model.eval()
        device=self.model.device
        max_len=60#for additional dataset, we don't generate too long
        max_len_a=20
        sos_b_id=self.sos_b_id
        eos_b_id=self.eos_b_id
        eos_a_id=self.eos_a_id
        with torch.no_grad():
            batches=[ original_batch ]
            new_batch=[]
            for batch in batches:
                try:
                    batch_size=len(batch)
                    #print('batch size:{}, turn_num:{}'.format(batch_size,len(batch[0])))
                    contexts=[[] for i in range(len(batch))]
                    resp=[[] for i in range(len(batch))]
                    aspn_gen=[[] for i in range(len(batch))]
                    bs_gen=[]
                    for turn_num in range(len(batch[0])):
                        past_key_values=None
                        end_flag=np.zeros(len(batch))
                        contexts=self.convert_eval_batch(batch,contexts,turn_num,bs_gen,\
                            prior=prior,resp_gen=resp,aspn_gen=aspn_gen)
                        inputs,attentions=self.reader.batch_align(contexts,left_len=max_len,return_attn=True)
                        inputs=torch.tensor(inputs).to(device)
                        attentions=torch.tensor(attentions).to(device)
                        if self.global_output>0 and cfg.example_log:
                            logging.info('generation examples:')
                            logging.info(self.tokenizer.decode(contexts[0]))
                            self.global_output-=1
                        for i in range(max_len):
                            position_ids = attentions.long().cumsum(-1) - 1
                            position_ids.masked_fill_(attentions == 0, 1)
                            if past_key_values is not None:
                                position_ids=position_ids[:, -1].unsqueeze(-1)
                            outputs=self.model(inputs,attention_mask=attentions,position_ids=position_ids,
                                return_dict=True,use_cache=True,past_key_values=past_key_values)#B,T,V
                            past_key_values=outputs.past_key_values
                            if cfg.sample_type=='top1':
                                preds=outputs.logits[:,-1,:].argmax(-1)#B
                            elif cfg.sample_type=='topk':
                                prob=F.softmax(outputs.logits[:,-1,:],dim=-1)#B,V
                                topk_probs, topk_words = torch.topk(prob, cfg.topk_num)#B,topk_num
                                widx = torch.multinomial(topk_probs, 1, replacement=True)#B,1
                                preds = torch.gather(topk_words, 1, widx).squeeze()#B
                            if i==0:
                                bs_tensor=preds.unsqueeze(1)
                            else:
                                bs_tensor=torch.cat([bs_tensor,preds.unsqueeze(1)],dim=1)
                            inputs=preds.unsqueeze(1)
                            attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(device)),dim=1)
                            end_flag+=(preds.cpu().numpy()==eos_b_id).astype(float)
                            if sum(end_flag==0)==0:
                                break
                        if cfg.gen_db:
                            bs_gen=self.get_bspn(bs_tensor,return_db=False,data=batch,turn_num=turn_num)
                            temp_contexts=self.convert_eval_batch(batch,contexts,turn_num,bs_gen,\
                                prior=prior,resp_gen=resp,aspn_gen=aspn_gen, gen_db=True)
                            inputs,attentions=self.reader.batch_align(temp_contexts,left_len=1,return_attn=True)
                            inputs=torch.tensor(inputs).to(device)
                            attentions=torch.tensor(attentions).to(device)
                            position_ids = attentions.long().cumsum(-1) - 1
                            position_ids.masked_fill_(attentions == 0, 1)

                            outputs=self.model(inputs,attention_mask=attentions,position_ids=position_ids,return_dict=True)
                            preds=outputs.logits[:,-1,:].argmax(-1)
                            db_gen=[]
                            for j in range(batch_size):
                                db_gen.append([self.sos_db_id, preds[j].item(),self.eos_db_id])
                        else:
                            bs_gen,db_gen=self.get_bspn(bs_tensor,return_db=True,data=batch,turn_num=turn_num)

                        contexts=self.convert_eval_batch(batch,contexts,turn_num,bs_gen,prior=prior,db_gen=db_gen)
                        if cfg.model_act:
                            past_key_values=None
                            end_flag=np.zeros(len(batch))
                            #note that the left_len should be max_len_a, but i set it to max_len to reduce the case of out of memory
                            inputs,attentions=self.reader.batch_align(contexts,left_len=max_len,return_attn=True)
                            inputs=torch.tensor(inputs).to(device)
                            attentions=torch.tensor(attentions).to(device)
                            if self.global_output>0 and cfg.example_log:
                                logging.info('generation examples:')
                                logging.info(self.tokenizer.decode(contexts[0]))
                            for i in range(max_len_a):
                                position_ids = attentions.long().cumsum(-1) - 1
                                position_ids.masked_fill_(attentions == 0, 1)
                                if past_key_values is not None:
                                    position_ids=position_ids[:, -1].unsqueeze(-1)
                                outputs=self.model(inputs,attention_mask=attentions,position_ids=position_ids,
                                    return_dict=True,use_cache=True,past_key_values=past_key_values)#B,T,V
                                past_key_values=outputs.past_key_values
                                if cfg.sample_type=='top1':
                                    preds=outputs.logits[:,-1,:].argmax(-1)#B
                                elif cfg.sample_type=='topk':
                                    prob=F.softmax(outputs.logits[:,-1,:],dim=-1)#B,V
                                    topk_probs, topk_words = torch.topk(prob, cfg.topk_num)#B,topk_num
                                    widx = torch.multinomial(topk_probs, 1, replacement=True)#B,1
                                    preds = torch.gather(topk_words, 1, widx).squeeze()#B
                                if i==0:
                                    bs_tensor=preds.unsqueeze(1)
                                else:
                                    bs_tensor=torch.cat([bs_tensor,preds.unsqueeze(1)],dim=1)
                                inputs=preds.unsqueeze(1)
                                attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(device)),dim=1)
                                end_flag+=(preds.cpu().numpy()==eos_a_id).astype(float)
                                if sum(end_flag==0)==0:
                                    break
                            aspn_gen=self.get_aspn(bs_tensor)

                        for i in range(len(batch)):
                            if validate:
                                batch[i][turn_num]['bspn_gen']=bs_gen[i]
                                batch[i][turn_num]['db_gen']=db_gen[i]
                                if cfg.model_act:
                                    batch[i][turn_num]['aspn_gen']=aspn_gen[i]
                            else:
                                batch[i][turn_num]['bspn']=bs_gen[i]
                                batch[i][turn_num]['db']=db_gen[i]
                                if cfg.model_act:
                                    batch[i][turn_num]['aspn']=aspn_gen[i]
                            if cfg.model_act:
                                resp[i]=batch[i][turn_num]['aspn']+batch[i][turn_num]['resp']#take aspn and resp as one resp
                            else:
                                resp[i]=batch[i][turn_num]['resp']
                    new_batch+=batch
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        logging.info("WARNING: ran out of memory during generation, and the batch will be divided half, batch size:{}, turn num:{}"\
                            .format(len(batch),len(batch[0])))
                        if hasattr(torch.cuda, 'empty_cache'):
                            with torch.cuda.device(device):
                                torch.cuda.empty_cache()
                        #current batch out of memory, split it half
                        batches+= [ batch[:len(batch)//2], batch[len(batch)//2:] ]
                    else:
                        logging.info(str(exception))
                        raise exception

        return new_batch

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
    
    def eval(self,data='dev',posterior=False,model=None):
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

    def load_data(self, path):
        data=json.load(open(path,'r', encoding='utf-8'))
        encoded_data=[]
        for dial in data:
            if cfg.len_limit and (len(data[dial])>16 or len(data[dial])<2):
                continue
            encoded_dial=[]
            max_len=0
            for turn in data[dial]:
                encoded_turn={}
                encoded_turn['dial_id']=dial
                encoded_turn['user']=self.tokenizer.encode('<sos_u>'+turn['user']+'<eos_u>')
                if cfg.delex_as_damd:
                    encoded_turn['resp']=self.tokenizer.encode('<sos_r>'+turn['resp_delex']+'<eos_r>')
                else:
                    encoded_turn['resp']=self.tokenizer.encode('<sos_r>'+turn['resp_delex1']+'<eos_r>')
                encoded_turn['turn_domain']=turn['turn_domain']
                encoded_dial.append(encoded_turn)
                user_len=len(encoded_turn['user'])
                resp_len=len(encoded_turn['resp'])
                if max(user_len,resp_len)>max_len:
                    max_len=max(user_len,resp_len)
            if cfg.len_limit and max_len>=76:
                continue
            encoded_data.append(encoded_dial)
        return encoded_data

    def validate_fast(self, data='dev', dial_id_list=None):
        # predict one dialog/ one turn at a time
        if cfg.mode=='pretrain' or cfg.mode=='train' or cfg.mode=='semi_ST':
            self.PrioriModel=self.model
            self.device1=self.model.device
        
        self.PrioriModel.eval()
        eval_data = self.reader.get_eval_data(data)
        
        if cfg.debugging:
            eval_data=eval_data[:100]
        if dial_id_list:
            new_data=[]
            for dial in eval_data:
                if dial[0]['dial_id']+'.json' in dial_id_list:
                    new_data.append(dial)
            eval_data=new_data
        
        cfg.batch_size=cfg.eval_batch_size
        batches=self.reader.get_batches('test',data=eval_data)
        result_path=os.path.join(cfg.eval_load_path,'result.json')
        
        if os.path.exists(result_path) and cfg.mode=='test' and not cfg.test_unseen_act and cfg.use_existing_result:
            if result_path[-3:]=='csv':
                results,field=self.reader.load_result(result_path)
            else:
                results=json.load(open(result_path,'r', encoding='utf-8'))
            joint_acc=compute_jacc(results)
            #joint_acc=0
            cfg.use_true_bspn_for_ctr_eval=True
            bleu, success, match, action_acc, P, R, F1 = self.evaluator.validation_metric(results, return_act_acc=True)
            score = 0.5 * (success + match) + bleu
            logging.info('validation [CTR] %2.2f  %2.2f  %2.2f  %.2f  %.3f' % (match, success, bleu, score, joint_acc))
            logging.info('Sys act P,R, F1: %.2f, %.2f, %.2f, %.2f' % (action_acc, P, R, F1))
            cfg.use_true_bspn_for_ctr_eval=False
            bleu, success, match = self.evaluator.validation_metric(results)
            score = 0.5 * (success + match) + bleu
            logging.info('validation %2.2f  %2.2f  %2.2f  %.2f  %.3f' % (match, success, bleu, score, joint_acc))

            eval_results = {}
            eval_results['bleu'] = bleu
            eval_results['success'] = success
            eval_results['match'] = match
            eval_results['score'] = score
            eval_results['joint_acc']=joint_acc
            eval_results['result'] = 'validation [CTR] match: %2.2f  success: %2.2f  bleu: %2.2f    score: %.2f' % (match, success, bleu, score)
            return eval_results
        
        

        # valid_losses = []
        result_collection = {}
        st=time.time()
        generated_data=[]
        for batch in batches:
            try:
                if batch==[]:
                    continue
                if cfg.turn_level:
                    batch=self.generate_batch_turn_level(batch)
                else:
                    batch=self.generate_batch_session_level(batch)
                for dialog in batch:
                    generated_data.append(dialog)
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
        if cfg.eval_resp_prob:
            avg_log_prob=self.compute_resp_prob(generated_data)
            logging.info('Avg log probability:{:.3f}'.format(avg_log_prob))
        cfg.use_true_bspn_for_ctr_eval=True
        bleu, success, match = self.evaluator.validation_metric(results)
        joint_acc=compute_jacc(results)
        #joint_acc=0
        score = 0.5 * (success + match) + bleu
        logging.info('validation [CTR] %2.2f  %2.2f  %2.2f  %.2f  %.3f' % (match, success, bleu, score, joint_acc))
        cfg.use_true_bspn_for_ctr_eval=False
        bleu, success, match, action_acc, P, R, F1 = self.evaluator.validation_metric(results, return_act_acc=True)
        score = 0.5 * (success + match) + bleu
        logging.info('validation %2.2f  %2.2f  %2.2f  %.2f  %.3f' % (match, success, bleu, score, joint_acc))
        logging.info('Sys act P,R, F1: %.2f, %.2f, %.2f, %.2f' % (action_acc, P, R, F1))
        
        #self.reader.save_result('w', results, field,result_name='result.csv')
        json.dump(results, open(result_path, 'w'), indent=2)

        eval_results = {}
        eval_results['bleu'] = bleu
        eval_results['success'] = success
        eval_results['match'] = match
        eval_results['score'] = score
        eval_results['joint_acc']=joint_acc
        eval_results['result'] = 'validation [CTR] match: %2.2f  success: %2.2f  bleu: %2.2f    \
            score: %.2f  joint goal:%.2f' % (match, success, bleu, score,joint_acc)
        cfg.batch_size=cfg.origin_batch_size
        return eval_results

    def compute_resp_prob(self, data):
        sys_model=GPT2LMHeadModel.from_pretrained('experiments_21/all_sys-model_sd11_lr0.0001_bs16_ga2/best_loss_model')
        # the tokenizer of sys_model should be the same as that of self.model
        all_batches, seq_num = self.reader.get_sys_batch(data, batch_size=16, mode='test')
        total_log_prob=0
        with torch.no_grad():
            for batch in all_batches:
                input_batch=torch.from_numpy(batch).long().to(sys_model.device)
                output_batch=sys_model(input_batch)
                loss = self.calculate_loss_and_accuracy(output_batch, input_batch)
                avg_log_prob=-loss.item()
                total_log_prob+=avg_log_prob
        return total_log_prob/len(all_batches)

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

    def generate_batch_session_level(self, batch):
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
                
                if self.global_output>0 and cfg.mode=='test':
                    logging.info(self.tokenizer.decode(contexts[0]))
                    self.global_output-=1
                
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
                pv_batch=self.reader.get_pv_batch(pv_batch, turn_batch['user'], resp_gen, bs_gen)
                turn_batch['bspn_gen']=bs_gen
                turn_batch['aspn_gen']=aspn_gen
                turn_batch['resp_gen']=resp_gen
                turn_batch['db_gen']=db_gen
        return self.reader.inverse_transpose_batch(batch)

    def validate_act(self,data='dev'):
        result_path=os.path.join(cfg.eval_load_path,'result.csv')
        if not os.path.exists(result_path):
            result_path=os.path.join(cfg.eval_load_path,'result.csv')

        if os.path.exists(result_path):
            results,field=self.reader.load_result(result_path)
            precision, recall, f1 = self.evaluator.aspn_eval(results)
            logging.info('Act Evaluation. P:{:.3f}, R:{:.3f}, F1:{:.3f}'.format(precision,recall,f1))
        else:
            logging.info('There is not existing result file')
            return None
    
    def validate_pos(self,data='dev'):
        result_path=os.path.join(cfg.eval_load_path,'result.csv')
        if os.path.exists(result_path) and 'test' in cfg.mode:
            results,field=self.reader.load_result(result_path)
            eval_result={}
            _, _, eval_result['act_F1']=self.evaluator.aspn_eval(results)
            eval_result['joint_acc'], eval_result['db_acc'] = compute_jacc(results,return_db=True)
            logging.info('Joint acc:{:.3f}, Act_F1:{:.3f}, DB_acc:{:.3f}'.\
                format(eval_result['joint_acc'],eval_result['act_F1'],eval_result['db_acc']))
            return eval_result
        eval_data = self.reader.get_eval_data(data)
        cfg.batch_size=cfg.eval_batch_size
        batches=self.reader.get_batches('test',data=eval_data)
        results=[]
        st=time.time()
        field = ['dial_id', 'user', 'resp', 'bspn', 'bspn_gen', 'aspn', 'aspn_gen', 'db','db_gen']
        for batch in batches:
            new_batch=self.gen_batch_bspn(batch,validate=True)
            for dial in new_batch:
                turn_temp={}
                for key in field:
                    if key=='dial_id':
                        turn_temp[key]=dial[0]['dial_id']
                    else:
                        turn_temp[key]=''
                results.append(turn_temp)
                for turn in dial:
                    entry={}
                    for key in field:
                        if key=='dial_id':
                            entry[key]=turn[key]
                        else:
                            entry[key]=self.tokenizer.decode(turn[key])
                    results.append(entry)
        logging.info('inference time:{:.3f} min'.format((time.time()-st)/60))
        self.reader.save_result('w', results, field,result_name='result.csv')
        eval_result={}
        _, _, eval_result['act_F1']=self.evaluator.aspn_eval(results)
        eval_result['joint_acc'], eval_result['db_acc'] = compute_jacc(results,return_db=True)
        logging.info('Joint acc:{:.3f}, Act_F1:{:.3f}, DB_acc:{:.3f}'.\
            format(eval_result['joint_acc'],eval_result['act_F1'],eval_result['db_acc']))
        return eval_result

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
        if cfg.turn_level:
            m.pretrain_turn_level()
    elif args.mode =='semi_VL':
        m.semi_VL()
    elif args.mode == 'semi_ST':
        m.semi_ST()
    elif args.mode =='test_all':
        pass

    elif args.mode =='test_pos':
        m.validate_pos(data='test')
    else:  # test
        logging.info('Load model from :{}'.format(cfg.eval_load_path))
        if cfg.test_unseen_act:
            dial_id_list=['pmul3647.json', 'mul2320.json', 'pmul4626.json', 'mul1650.json', 
            'pmul0286.json', 'mul2491.json', 'pmul4356.json', 'mul2637.json', 'mul0690.json', 
            'pmul1180.json', 'pmul4660.json', 'pmul2942.json', 'pmul3672.json', 'mul0810.json', 
            'pmul0182.json', 'mul1546.json', 'pmul1521.json', 'pmul1966.json', 'mul1008.json', 
            'pmul1247.json', 'mul0309.json', 'pmul3737.json', 'mul0818.json', 'pmul4884.json', 
            'mul0004.json', 'sng0840.json', 'sng1042.json', 'mul1901.json', 'pmul4316.json', 
            'pmul3145.json', 'pmul0599.json', 'mul0374.json', 'sng0994.json', 'sng0636.json', 
            'pmul2009.json', 'pmul4542.json', 'mul0772.json', 'sng0691.json', 'mul0822.json', 
            'sng0690.json', 'pmul4362.json', 'pmul3439.json', 'mul2137.json', 'mul2658.json', 
            'mul1527.json', 'pmul4050.json', 'mul0397.json', 'pmul4155.json', 'mul1211.json', 
            'sng0897.json', 'pmul0864.json', 'mul1766.json', 'mul0089.json', 'mul0901.json',
            'mul0941.json', 'pmul4504.json', 'pmul1470.json', 'mul0738.json', 'mul0080.json', 
            'mul0739.json', 'mul2525.json', 'pmul0844.json', 'pmul0090.json', 'pmul3933.json', 
            'pmul3107.json', 'pmul3897.json', 'sng0979.json', 'mul1268.json', 'pmul4911.json', 
            'mul1678.json', 'pmul4357.json', 'pmul4247.json', 'mul1848.json', 'sng0081.json', 
            'pmul1537.json', 'pmul3066.json', 'sng01673.json', 'mul1077.json', 'mul0034.json', 
            'mul2009.json', 'mul2270.json', 'mul2482.json', 'mul1649.json', 'mul0113.json', 
            'pmul4220.json', 'pmul4224.json', 'pmul4246.json', 'sng0589.json', 'pmul3423.json', 
            'pmul3778.json', 'pmul4234.json', 'mul2439.json', 'mul1661.json', 'mul2099.json', 
            'mul0760.json', 'pmul2578.json', 'mul2197.json', 'pmul0998.json', 'mul1071.json', 
            'mul1285.json', 'mul2423.json', 'pmul3520.json', 'pmul0615.json', 'mul0533.json', 
            'pmul0441.json', 'mul1254.json', 'pmul3044.json', 'mul1935.json', 'mul0466.json', 
            'mul2042.json', 'sng1105.json', 'mul0841.json', 'pmul2146.json', 'pmul1194.json', 
            'pmul4716.json', 'pmul4644.json', 'sng02319.json', 'pmul4547.json']
            m.validate_fast('test', dial_id_list=dial_id_list)
        else:
            m.validate_fast('test')


if __name__ == "__main__":
    main()
