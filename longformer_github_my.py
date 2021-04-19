import torch
from longformer.longformer import Longformer, LongformerConfig
from longformer.sliding_chunks import pad_to_window_size
from transformers import RobertaTokenizer, LongformerTokenizer

config = LongformerConfig.from_pretrained('pre-model/' + 'longformer-encdec-base-16384')
# choose the attention mode 'n2', 'tvm' or 'sliding_chunks'
# 'n2': for regular n2 attantion
# 'tvm': a custom CUDA kernel implementation of our sliding window attention
# 'sliding_chunks': a PyTorch implementation of our sliding window attention
config.attention_mode = 'sliding_chunks'

model = Longformer.from_pretrained('pre-model/' + 'longformer-encdec-base-16384', config=config)
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
tokenizer = LongformerTokenizer.from_pretrained('pre-model/' + 'longformer-encdec-base-16384')
tokenizer.model_max_length = model.config.max_position_embeddings

t1 = "ATGCATGCCGTAATAACGTC"  # long input document
t2 = "AGTCACGTTGCA"  # long input document

tokenize = tokenizer.tokenize(t1)
print(tokenize)
input_ids = tokenizer(t1, return_attention_mask=True, return_token_type_ids=True)
print(input_ids)
# TVM code doesn't work on CPU. Uncomment this if `config.attention_mode = 'tvm'`
# model = model.cuda(); input_ids = input_ids.cuda()

# # Attention mask values -- 0: no attention, 1: local attention, 2: global attention
# attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)  # initialize to local attention
# attention_mask[:, [1, 4, 21, ]] = 2  # Set global attention based on the task. For example,
# # classification: the <s> token
# # QA: question tokens

# # padding seqlen to the nearest multiple of 512. Needed for the 'sliding_chunks' attention
# input_ids, attention_mask = pad_to_window_size(
#     input_ids, attention_mask, config.attention_window[0], tokenizer.pad_token_id)

output = model(**input_ids)[0]

print(output)
print(output.shape)
