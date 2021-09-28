from config import global_config as cfg
from eval import MultiWozEvaluator
from reader import MultiWozReader
from transformers import GPT2Tokenizer
import json
import ontology

tokenizer=GPT2Tokenizer.from_pretrained('best_model')
reader = MultiWozReader(tokenizer)
evaluator = MultiWozEvaluator(reader)

bspn="<sos_b> [restaurant]  food modern european pricerange moderate <eos_b>"
domain='restaurant'
constraint_dict = reader.bspan_to_constraint_dict(bspn)
print('constraint_dict\n',constraint_dict)
venues = reader.db.queryJsons(domain, constraint_dict[domain], return_name=True)
print('venues:', venues)
