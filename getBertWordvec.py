import torch
from pytorch_transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('E:/bert/6-new-12w-0')
model = BertModel.from_pretrained('E:/bert/6-new-12w-0')
input_ids = torch.tensor(tokenizer.encode("自然语" * 1000)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)
# last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
sequence_output = outputs[0]
pooled_output = outputs[1]
print(sequence_output)
print(sequence_output.shape)  ## 字向量
print(pooled_output.shape)  ## 句向量
