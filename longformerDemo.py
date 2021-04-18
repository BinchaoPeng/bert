from transformers import AutoModel, AutoTokenizer, pipeline
import torch

model_name = 'pre-model/' + 'longformer-encdec-base-16384'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
classifier = pipeline('feature-extraction', model=model, tokenizer=tokenizer)

# encoded_inputs = tokenizer(["ATGCATGCNACT"], ["ATGCATGCNACT"], return_token_type_ids=True, return_tensors='pt')
encoded_inputs = tokenizer(["ATGCATGCNACT", "ATGCATG", "ACTGGTCATGCAC"], return_tensors='pt',
                           padding=True)
print(encoded_inputs)
# feature = model(input_ids=encoded_inputs['input_ids'], attention_mask=encoded_inputs['attention_mask'],
#                 return_netsors='pt')
feature = model(**encoded_inputs,
                return_netsors='pt')
print(feature[0])
print(type(feature[0]))
# feature = torch.as_tensor(feature)
# print(feature.shape)
print("***" * 48)

feature = classifier(["ATG", "ATGCATG", "ACTGGTCATGCAC"])
print(type(feature))
feature = torch.as_tensor(feature)
print(feature)
print(feature.shape)
print("***" * 48)
