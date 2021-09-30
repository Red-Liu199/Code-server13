from config import global_config as cfg
from eval import MultiWozEvaluator
from reader import MultiWozReader
from transformers import GPT2Tokenizer
import json
import ontology

tokenizer=GPT2Tokenizer.from_pretrained('experiments_21/turn-level-DS/best_score_model')
reader = MultiWozReader(tokenizer)
evaluator = MultiWozEvaluator(reader)
goal={
      "taxi": {
        "info": {
          "leaveat": "20:00"
        },
        "reqt": [
          "car",
          "phone"
        ],
        "fail_info": {}
      },
      "hotel": {
        "info": {
          "area": "centre",
          "pricerange": "moderate",
          "internet": "yes"
        },
        "fail_info": {},
        "book": {
          "pre_invalid": True,
          "people": "3",
          "day": "sunday",
          "invalid": False,
          "stay": "3"
        },
        "fail_book": {}
      },
      "restaurant": {
        "info": {
          "food": "chinese",
          "pricerange": "moderate",
          "area": "centre"
        },
        "reqt": [
          "address",
          "phone",
          "postcode"
        ],
        "fail_info": {}
      }
    }
goal=reader.goal_norm(goal)
print(goal)
print(reader.goal_to_gpan(goal))