import numpy as np
from transformers import LongformerModel, LongformerTokenizer, LongformerConfig

model_name = 'pre-model/' + 'longformer-base-4096'
config = LongformerConfig.from_pretrained(model_name)
tokenizer = LongformerTokenizer.from_pretrained(model_name)
model = LongformerModel.from_pretrained(model_name, config=config)

txts = ["ATGCAGTA", "ATGCATGCA", "ACGTACGATGCAAA"]

max = 0
for txt in txts:
    tokens = tokenizer.tokenize(txt)
    token_value = len(tokens)
    print(tokens)
    print(token_value)

    if token_value > max:
        max = token_value

txts1 = ["ATGCA", "ACTGACGTA", "ACG"]

encoded_inputs = tokenizer(txts, return_tensors='pt', padding=True,
                           max_length=max)
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
