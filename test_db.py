from config import global_config as cfg
from eval import MultiWozEvaluator
from reader import MultiWozReader
from transformers import GPT2Tokenizer
import json
import ontology

tokenizer=GPT2Tokenizer.from_pretrained('experiments_21/turn-level-DS/best_score_model')
reader = MultiWozReader(tokenizer)
evaluator = MultiWozEvaluator(reader)
user_act=' [attraction] [request]  fee address phone [attraction] [inform]  name cambridge artworks'
print(reader.aspan_to_act_dict(user_act, side='user'))