import torch
from pytorch_transformers import BertTokenizer, BertModel
import platform

# system = str(platform.system())
# print(system)
# if (system == "Windows"):
#     url = 'E:/bert/'
# elif (system == "Linux"):
#     url = 'pre-model1/'
# else:
#     print("Other System tasks")

model_name = 'pre-model/' + '6-new-12w-0'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

input_ids = torch.tensor(tokenizer.encode("[CLS] ATGTTC TATGAG [SEP] ATGTTC TATGAG " * 500)).unsqueeze(
    0)  # Batch size 1

print(input_ids)
outputs = model(input_ids)
# last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
sequence_output = outputs[0]
pooled_output = outputs[1]
print(sequence_output)
print(sequence_output.shape)  ## 字向量
print(pooled_output)
print(pooled_output.shape)  ## 句向量
