import torch
from pytorch_transformers import BertTokenizer, BertModel
import platform

model_name = 'pre-model/' + '6-new-12w-0'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

sep = tokenizer.tokenize("ATGC")

print(sep)
