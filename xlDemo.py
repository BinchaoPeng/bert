import numpy as np
from transformers import TransfoXLModel, TransfoXLTokenizer, TransfoXLConfig, TransfoXLCorpus

model_name = 'pre-model/' + 'transfo_xl_wt103'
config = TransfoXLConfig.from_pretrained(model_name)
tokenizer = TransfoXLTokenizer.from_pretrained(model_name)
model = TransfoXLModel.from_pretrained(model_name, config=config)

txts = ["ATGCAGTA", "ATGCATGCA", "ACGTACGATGCAAA"]
txts1 = ["ATGCA", "ACTGACGTA", "ACG"]

max = 0
for txt in txts:
    tokens = tokenizer.tokenize(txt)
    token_value = len(tokens)
    print(tokens)
    print(token_value)

    if token_value > max:
        max = token_value



encoded_inputs = tokenizer(txts, txts1, return_tensors='pt')
print(encoded_inputs)

for input_id in encoded_inputs['input_ids']:
    seq = tokenizer.decode(input_id)
    print(seq)

X_enpr_features = model(**encoded_inputs)[0]
print(X_enpr_features.shape)

X_enpr = X_enpr_features.detach().numpy()

print(X_enpr)
print(X_enpr.size)
print(X_enpr.shape)
# encoded_inputs = tokenizer(txts1, return_tensors='pt', padding=True,
#                            max_len=max)
# X_enpr_features = model(**encoded_inputs)
# X_enpr = np.array(X_enpr_features)

# print("output:")
# for item in X_enpr:
#     print(item)
#     print("\n")
#
# np.savez("test.npz", x1=X_enpr, x2=X_enpr)
#
# print("saved!")
