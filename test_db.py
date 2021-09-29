from config import global_config as cfg
from eval import MultiWozEvaluator
from reader import MultiWozReader
from transformers import GPT2Tokenizer
import json
import ontology

tokenizer=GPT2Tokenizer.from_pretrained('best_model')
reader = MultiWozReader(tokenizer)
evaluator = MultiWozEvaluator(reader)

aspn="[hotel] [inform] parking yes price cheap name guest a and b house [request] internet"
act=reader.aspan_to_act_dict(aspn, side='sys')
print(act)
act=reader.aspan_to_act_dict(aspn, side='user')
print(act)