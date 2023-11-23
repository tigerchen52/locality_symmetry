import torch
import torch.nn as nn



# a = torch.tensor([[1, 1, 0], [1, 0, 0]]).bool()
# print(a)
# print(a.size())
# b = a.unsqueeze(-1)
# print(b.size())
# print(b)
# c = b.repeat(1,1,5)
# print(c.size())
# print(c)
#
# value = torch.rand((2,3,5))
# print(value)
# print(value.masked_fill_(~c, 0.))

# a = torch.tensor([[1,  0], [0, 1]])
# c = a.max(1)[1]
# print(c)
# b = torch.tensor([0,1])
# d = (b==c).int()
# print(d.numpy())
# import re
# a = '^MIter (loss=56.163): : 0it [00:01, ?it/s]^MIter (loss=56.163): : 1it [00:01,  1.49s/it]^MIter (loss=56.458): : 1it [00:02,  1.49s/it]^MIter (loss=56.458): '
# p = re.compile('(?<=Iter \(loss=)\d+\.?\d*')
# result = p.findall(a)
# print(result)

# record = {
#     'CoLA': [0.2395164213440934, 0.1964367401170084, 0.14369673835432495, 0.11304788052863822, 0.0],
#     'SST-2': [0.8153669238090515, 0.8188073039054871, 0.7993118762969971, 0.7912843823432922, 0.7373852729797363],
#     'MRPC':  [0.6904347538948059, 0.7008695602416992, 0.7118840217590332, 0.7043477892875671, 0.6962318420410156],
#     'QQP': [0.8004699349403381, 0.7948800325393677, 0.7758842706680298, 0.7636408805847168, 0.7259708046913147],
#     'STS-B': [0.780749148233925, 0.7969411898386238, 0.8011011055171162, 0.8002576245087243, 0.7852372476083107],
#     'MNLI-m':[0.643810510635376, 0.63270503282547, 0.6196637749671936, 0.604788601398468, 0.5724911093711853],
#     'MNLI-mm':[0.6544955372810364, 0.644934892654419, 0.6346623301506042, 0.6251016855239868, 0.5925549268722534],
#     'QNLI':  [0.7986454367637634, 0.8015742301940918, 0.7918725609779358, 0.7849166989326477, 0.7721032500267029],
#     'RTE': [0.6101083159446716, 0.6209385991096497, 0.5992779731750488, 0.5992779731750488, 0.577617347240448],
#     'WNLI':  [0.35211268067359924, 0.4507042169570923, 0.4647887349128723, 0.4647887349128723, 0.5352112650871277]
# }
#
# lrs = ['3e-4', '1e-4', '5e-5', '3e-5', '1e-5']
# w_l = ''
# prefix = 'PAM-G&140K&'
# for k, values in record.items():
#     w_l += prefix + k
#     for i, v in enumerate(values):
#        w_l += '&' + lrs[i] + '&' + str(round(v*100, 2))
#
#     w_l += '&' + str(round(max(values) *100, 2)) + '\cr\n'
#
# print(w_l)
import nltk
w_l = ''
with open('data/wiki21.csv', encoding='utf8')as f:
    lines = f.readlines()
    for line in lines:
        doc = line.strip().split(',')[1]
        for sent in nltk.sent_tokenize(doc):
            w_l += sent + '\n'
        w_l += '\n'

with open('data/evaluation.txt', 'w', encoding='utf8')as f:
    f.write(w_l)