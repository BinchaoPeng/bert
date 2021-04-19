from transformers import AutoModel, AutoTokenizer, pipeline
import torch

model_name = 'pre-model/' + 'longformer-encdec-base-16384'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
classifier = pipeline('feature-extraction', model=model, tokenizer=tokenizer)

txtList = ["ATGCACGTAC" * 300 + tokenizer.sep_token + "CGTAGCATCG" * 200,
           "ATCGGCTACT" * 300 + tokenizer.sep_token + "GCATGCATGC" * 200,
           "TCAGAGACTG" * 300 + tokenizer.sep_token + "ACGTGCNACT" * 200,
           "GGGTCANACT" * 300 + tokenizer.sep_token + "TTTCGAACCT" * 200
           ]

# encoded_inputs = tokenizer(["ATGCATGCNACT"], ["ATGCATGCNACT"], return_token_type_ids=True, return_tensors='pt')
encoded_inputs = tokenizer(txtList, return_tensors='pt', padding=True)
print(encoded_inputs)
print("***" * 48)

for item in encoded_inputs['input_ids']:
    decoder = tokenizer.decode(item)
    print(decoder)
print("***" * 48)

feature = model(**encoded_inputs,
                return_netsors='pt')
print(feature[0])
print(type(feature[0]))
# feature = torch.as_tensor(feature)
# print(feature.shape)
print("***" * 48)

# feature = classifier(txtList)
# print(type(feature))
# feature = torch.as_tensor(feature)
# print(feature)
# print(feature.shape)
# print("***" * 48)
