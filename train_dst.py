from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer,BertLMHeadModel, BertConfig
from bert_reader import Bert_Reader
import torch
import os
import random
import argparse
import time
import logging
import json
import tqdm
import numpy as np
from config import global_config as cfg
from compute_joint_acc import compute_jacc
from torch.utils.tensorboard import SummaryWriter

class BERT_DST(object):
    def __init__(self):
        config = BertConfig.from_pretrained("bert-base-uncased")
        config.is_decoder = True
        self.model=BertLMHeadModel.from_pretrained('bert-base-uncased',config=config)
        self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
        self.reader=Bert_Reader(self.tokenizer)
        self.model.resize_token_embeddings(len(self.tokenizer))
        if len(cfg.cuda_device)>1:
            self.device=cfg.cuda_device
        else:
            self.device=cfg.cuda_device[0]
        self.model.to(self.device)
        self.tb_writer=SummaryWriter(log_dir=os.path.join('./dst_log',cfg.exp_no))
        json.dump(cfg.__dict__,open(os.path.join(cfg.exp_path,'cfg_all.json'),'w'),indent=2)

    def get_optimizers(self,dial_nums):
       
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)
        num_training_steps = dial_nums*cfg.epoch_num // (cfg.gradient_accumulation_steps*cfg.batch_size)
        num_warmup_steps = cfg.warmup_steps if cfg.warmup_steps >= 0 else int(num_training_steps*0.2)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def show_examples(self,inputs,labels):
        #inputs:[[],[],...,[]]
        #labels:[[],[],...,[]]
        for text1,text2 in zip(inputs[:2],labels[:2]):
            if cfg.pad_id in text1:
                logging.info('input sample:{}'.format(self.tokenizer.decode(text1[:text1.index(cfg.pad_id)])))
            else:
                logging.info('input sample:{}'.format(self.tokenizer.decode(text1)))
            if -100 in text2:
                logging.info('label sample:{}'.format(self.tokenizer.decode(text2[:text2.index(-100)])))
            else:
                logging.info('label sample:{}'.format(self.tokenizer.decode(text2)))

    def eval(self,data='dev',test=False):
        all_batches=self.reader.get_batches(data)
        sos_b_id=self.tokenizer.convert_tokens_to_ids('<sos_b>')
        eos_b_id=self.tokenizer.convert_tokens_to_ids('<eos_b>')
        eval_data=[]
        self.model.eval()
        with torch.no_grad():
            for batch_idx, dial_batch in enumerate(all_batches):
                turn_nums=[len(dial) for dial in dial_batch]
                batch_size=len(dial_batch)
                for i in range(turn_nums[0]):
                    
                    inputs,labels=self.reader.convert_batch_turn(dial_batch,i)

                    inputs['input_ids']=torch.tensor(inputs['input_ids']).to(self.device)
                    inputs['attention_mask']=torch.tensor(inputs['attention_mask']).to(self.device)
                    outputs=self.model(**inputs)
                    logits=outputs[0]#B,T,V
                    preds=logits.argmax(-1).tolist()#B,T
                    if test and batch_idx==0:
                        logging.info(preds)
                    for j in range(batch_size):
                        turn_entry={}
                        bspn_gen=preds[j]
                        turn_entry['user']=dial_batch[j][i]['user']
                        turn_entry['bspn']=dial_batch[j][i]['bspn']
                        if sos_b_id in bspn_gen:
                            bspn_gen=bspn_gen[bspn_gen.index(sos_b_id)+1:]
                        if eos_b_id in bspn_gen:
                            bspn_gen=bspn_gen[:bspn_gen.index(eos_b_id)]

                        turn_entry['bspn_gen']=self.tokenizer.decode(bspn_gen)
                        turn_entry['turn_num']=i
                        eval_data.append(turn_entry)
        joint_acc=compute_jacc(eval_data)
        return joint_acc

    def save_model(self):
        save_path=os.path.join(cfg.exp_path, 'best_model')
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)


    def train(self):

        all_batches,dial_nums,turn_nums = self.reader.get_batches('train',return_nums=True)
        optimizer, scheduler = self.get_optimizers(dial_nums)
        logging.info("***** Running training *****")
        logging.info("  Num Turns = %d", turn_nums)
        logging.info("  Num Dialogs = %d", dial_nums)
        logging.info("  Num Epochs = %d", cfg.epoch_num)
        logging.info("  Batch size  = %d", cfg.batch_size)
        logging.info("  Gradient Accumulation steps = %d",cfg.gradient_accumulation_steps)
        
        log_samples=1
        max_acc=0
        early_stop_count=cfg.early_stop_count
        st=time.time()
        joint_acc=self.eval()
        logging.info('Initial joint goal:{}, evaluation time:{}'.format(joint_acc,time.time()-st))
        global_step=0

        for epoch in range(cfg.epoch_num):
            epoch_step = 0
            tr_loss = 0.0
            logging_loss = 0.0
            btm = time.time()
            oom_time = 0
            self.model.zero_grad()
            random.shuffle(all_batches)
            for batch_idx, dial_batch in enumerate(all_batches):
                self.model.train()
                try:
                    all_turn_nums=[len(dial) for dial in dial_batch]#当前batch的对话轮数
                    assert all([turn_num==all_turn_nums[0] for turn_num in all_turn_nums])
                    for i in range(all_turn_nums[0]):
                        inputs,labels=self.reader.convert_batch_turn(dial_batch,i)
                        if log_samples>0:
                            self.show_examples(inputs['input_ids'],labels['input_ids'])
                            log_samples-=1
                        inputs['input_ids']=torch.tensor(inputs['input_ids']).to(self.device)
                        inputs['attention_mask']=torch.tensor(inputs['attention_mask']).to(self.device)
                        labels['input_ids']=torch.tensor(labels['input_ids']).to(self.device)
                        outputs=self.model(**inputs,return_dict=True,labels=labels['input_ids'])
                        loss=outputs.loss
                        loss.backward()
                        tr_loss+=loss.item()
                        epoch_step+=1
                        if epoch_step%cfg.gradient_accumulation_steps==0 or epoch_step==turn_nums-1:
                            optimizer.step()
                            scheduler.step()
                            global_step+=1

                            loss_scalar=tr_loss-logging_loss
                            logging_loss=tr_loss
                            self.tb_writer.add_scalar('loss',loss_scalar,global_step)

                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        oom_time+=1
                        logging.info("WARNING: ran out of memory, times:{},batch idx:{}, batch size:{}"\
                            .format(oom_time,batch_idx,len(dial_batch)))
                    else:
                        logging.info(str(exception))
                        raise exception
            if epoch==0:
                joint_acc=self.eval(test=True)
            else:
                joint_acc=self.eval()
            logging.info('Epoch:{}, time:{}, joint goal:{}'.format(epoch, time.time()-btm, joint_acc))
            if joint_acc>max_acc:
                self.save_model()
                max_acc=joint_acc
                early_stop_count=cfg.early_stop_count
            elif cfg.early_stop:
                early_stop_count-=1
                if early_stop_count==0:
                    logging.info('early stopped')
                    break

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


def main():
    if not os.path.exists('./experiments'):
        os.mkdir('./experiments')

    if not os.path.exists('./experiments_21'):
        os.mkdir('./experiments_21')

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()
    parse_arg_cfg(args)
    cfg.mode = args.mode

    cfg._init_logging_handler(args.mode)
    experiments_path = './experiments'
    cfg.exp_path = os.path.join(experiments_path,cfg.exp_no)
    if not os.path.exists(cfg.exp_path):
        os.mkdir(cfg.exp_path)

    # fix random seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    m = BERT_DST()

    if args.mode=='train':
        m.train()
    else:  # test
        pass


if __name__ == "__main__":
    main()


