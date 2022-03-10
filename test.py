import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel


if __name__=='__main__':
    #加载预训练模型
    tokenizer = GPT2Tokenizer.from_pretrained('../distilgpt2')
    model = GPT2LMHeadModel.from_pretrained('../distilgpt2')

    device='cpu'#'cuda:0'
    model.to(device)

    #对输入语句进行编码
    inputs = tokenizer.encode("Hello, my dog is cute")
    length=20 # 生成的长度
    #自回归生成，可以将其中的概率保存起来
    for i in range(length):
        outputs=model(torch.tensor([inputs]).to(device))

        # logits and probabilities, shape: 1*len*vocab
        logits=outputs[0]
        probs=F.softmax(logits, dim=-1)
        # 当前token的概率，可用于校准
        prob=probs[:, -1, :]

        # greedy search
        next_token_id=logits[0, -1, :].argmax().item()
        inputs+=[next_token_id]

    #输出生成结果
    print(tokenizer.decode(inputs))
