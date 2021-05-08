import torch
from transformers import BertModel, BertConfig, DNATokenizer

model_name = 'pre-model/' + '6-new-12w-0'

config = BertConfig.from_pretrained(model_name)
tokenizer = DNATokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, config=config)

sequence = "AATCTA ATCTAG TCTAGC CTAGCA"
model_input = tokenizer.encode_plus(sequence, add_special_tokens=True, max_length=512)["input_ids"]
model_input = torch.tensor(model_input, dtype=torch.long)
model_input = model_input.unsqueeze(0)  # to generate a fake batch with batch size one

output = model(model_input)
print(output[0])

print(output[0].shape, "\n\n")
# CLS token's embedding

print(output[1])
print(output[1].shape, "\n\n")
