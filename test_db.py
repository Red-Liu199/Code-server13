from config import global_config as cfg
from eval import MultiWozEvaluator
from reader import MultiWozReader
from transformers import GPT2Tokenizer
import json
import ontology

tokenizer=GPT2Tokenizer.from_pretrained('experiments_21/turn-level-DS/best_score_model')
reader = MultiWozReader(tokenizer)
evaluator = MultiWozEvaluator(reader)
user_act='[train] [request]  id'
goal0='[hotel] [inform] internet yes stars 4 area north [book] stay 3 day sunday people 5 [train] [request] id'
print(reader.aspan_to_act_dict(user_act, side='user'))
bspn="<sos_b> [train] [inform]  price price price price price price price price price price price price price price price price price price price price price price price price price price price price price price price price price price <eos_a> <eos_a> <sos_r> [value_price] [value_price] [value_price] [value_price] [value_price] [value_price] [value_price] [value_price] [value_price] [value_price] <eos_b>"
print(reader.bspan_to_constraint_dict(bspn))