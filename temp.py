from transformers import GPT2Tokenizer, GPT2Model
import torch
import os, json

if __name__=='__main__':
    '''
    tokenizer = GPT2Tokenizer.from_pretrained('../distilgpt2')
    model = GPT2Model.from_pretrained('../distilgpt2')

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    outputs = model(**inputs, output_attentions=True, return_dict=True)
    print(outputs['attentions'][0][0][0])
    '''
    dir='data/multi-woz-2.1-processed'
    dial_list=[]
    for ratio in range(5, 30, 5):
        path=os.path.join(dir, 'divided_data%d.json'%ratio)
        if os.path.exists(path):
            print(path)
            data=json.load(open(path, 'r'))
            dial_ids=[]
            for dial  in data['pre_data']:
                dial_ids.append(dial[0]['dial_id'])
            dial_list.append(dial_ids)
    for i in range(len(dial_list)-1):
        dials1, dials2 = dial_list[i], dial_list[i+1]
        count=0
        for dial in dials1:
            if dial not in dials2:
                count+=1
        print('%d not in %d: %d'%((i+1)*5, (i+2)*5, count))
        

            
