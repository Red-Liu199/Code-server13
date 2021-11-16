import json
import numpy as np
import math
import scipy.stats as st
def compute_P(N10, K):
    w=(abs(N10-K/2)-0.5)/math.sqrt(K/4)
    P=2*(1-st.norm(0,1).cdf(abs(w)))
    return P

def matched_pair(list1, list2):
    n=len(list1)
    Z=[item1-item2 for item1, item2 in zip(list1, list2)]
    u=np.mean(Z)
    sigma=math.sqrt(sum([(z-u)**2 for z in Z])/(n-1))
    w=u*math.sqrt(n)/sigma
    P=2*(1-st.norm(0,1).cdf(abs(w)))
    return P

path0='/home/liuhong/myworkspace/best_model/'
path1='/home/liuhong/myworkspace/experiments_21/DS_base/best_score_model/'
inform0=json.load(open(path0+'match_list.json', 'r', encoding='utf-8'))
inform1=json.load(open(path1+'match_list.json', 'r', encoding='utf-8'))
success0=json.load(open(path0+'success_list.json', 'r', encoding='utf-8'))
success1=json.load(open(path1+'success_list.json', 'r', encoding='utf-8'))
IN00, IN01, IN10, IN11 = 0,0,0,0
SN00, SN01, SN10, SN11 = 0,0,0,0
IN00=sum([i1 and i2 for i1, i2 in zip(inform0, inform1)])
IN01=sum([i1 and not i2 for i1, i2 in zip(inform0, inform1)])
IN10=sum([not i1 and i2 for i1, i2 in zip(inform0, inform1)])
IN11=sum([not i1 and not i2 for i1, i2 in zip(inform0, inform1)])
SN00=sum([i1 and i2 for i1, i2 in zip(success0, success1)])
SN01=sum([i1 and not i2 for i1, i2 in zip(success0, success1)])
SN10=sum([not i1 and i2 for i1, i2 in zip(success0, success1)])
SN11=sum([not i1 and not i2 for i1, i2 in zip(success0, success1)])
print(dict(IN00=IN00, IN01=IN01, IN10=IN10, IN11=IN11))
print(dict(SN00=SN00, SN01=SN01, SN10=SN10, SN11=SN11))
IK=IN10+IN01
SK=SN10+SN01
IP=compute_P(IN10, IK)
SP=compute_P(SN10, SK)
print(dict(IP=IP, SP=SP))
bleu0=json.load(open(path0+'bleu_list.json', 'r', encoding='utf-8'))
bleu1=json.load(open(path1+'bleu_list.json', 'r', encoding='utf-8'))
BP=matched_pair(bleu0, bleu1)
print(dict(BP=BP))
BP=matched_pair(inform0, inform1)
print(dict(BP=BP))
BP=matched_pair(success0, success1)
print(dict(BP=BP))
