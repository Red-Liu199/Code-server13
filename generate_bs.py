from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
from eval import MultiWozEvaluator
from damd_net import DAMD, cuda_, get_one_hot_input
from reader import MultiWozReader
import utils
from torch.optim import Adam
import torch
import torch.nn as nn

import os
import random
import argparse
import time
import logging
import json
import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

#修改记录：config与config21按情况切换，reader.py里也一样
from config import global_config as cfg 
#from config21 import global_config as cfg  # global, already initialized




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
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()
    parse_arg_cfg(args)

    cfg._init_logging_handler(args.mode)

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)


    tokenizer = GPT2Tokenizer.from_pretrained(cfg.eval_load_path)
    reader = MultiWozReader(tokenizer)
    model= GPT2LMHeadModel.from_pretrained(cfg.eval_load_path)
    device = torch.device("cuda:{}".format(cfg.cuda_device[0]))
    model=model.to(device)
    model.eval()
    eval_data = reader.get_eval_data('train')
    #generate the rest half
    eval_data=eval_data[len(eval_data)//2:]
    logging.info('data nums:%d'%len(eval_data))
    log_inputs=5
    max_len=50
    truncated_count=0
    with torch.no_grad():
        for dial_idx, dialog in enumerate(tqdm(eval_data)):
            pv_context=[]
            for turn_idx, turn in enumerate(dialog):
                input_context=pv_context+turn['user']+turn['aspn']+turn['resp']+[tokenizer.convert_tokens_to_ids('<sos_b>')]
                context_length=len(input_context)
                if context_length>1024-max_len:
                    #后截断
                    logging.info('over length, context:%s'%tokenizer.decode(input_context))
                    input_context=input_context[max_len-1024:]
                    truncated_count+=1
                    context_length=1024-max_len
                input_tensor=torch.tensor([input_context]).long().to(device)
                if log_inputs > 0:
                    logging.info(tokenizer.decode(input_context))
                    log_inputs -= 1
                outputs = model.generate(input_ids=input_tensor,
                                                    max_length=context_length+max_len, temperature=0.7, #top_p=0.9, num_beams=4,
                                                    pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.encode(['<eos_b>'])[0])
                generated_bs = outputs[0].cpu().numpy().tolist()
                generated_bs=generated_bs[context_length-1:]
                eos_b_id = tokenizer.encode(['<eos_b>'])[0]
                if eos_b_id in generated_bs:
                    eos_b_idx = generated_bs.index(eos_b_id)
                else:
                    eos_b_idx = len(generated_bs)-1
                bspn_gen=generated_bs[: eos_b_idx+1]
                db_result = reader.bspan_to_DBpointer(tokenizer.decode(bspn_gen), turn['turn_domain'])
                db = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('<sos_db> '+ db_result + ' <eos_db>'))
                if log_inputs>0:
                    logging.info('generated belief state:{}'.format(tokenizer.decode(bspn_gen)))
                    logging.info('original belief state:{}'.format(tokenizer.decode(turn['bspn'])))
                turn['bspn']=bspn_gen
                turn['db']=db
                pv_context+=turn['user']+turn['aspn']+turn['resp']+turn['bspn']
                if log_inputs>0:
                    #logging.info('generated belief state again:{}'.format(tokenizer.decode(eval_data[dial_idx][turn_idx]['bspn'])))
                    logging.info('db from generated belief state: %s'%tokenizer.decode(db))
    encoded_data={'half_train':eval_data}
    json.dump(encoded_data, open('/home/liuhong/UBAR-MultiWOZ/data/multi-woz-processed/half_train_data.json', 'w'), indent=2)

if __name__ == "__main__":
    main()