import torch
from transformers import BertModel, BertConfig, BertTokenizer  # , DNATokenizer

model_name = 'pre-model/' + '6-new-12w-0'

# config = BertConfig.from_pretrained(
#     'https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-6/config.json')
config = BertConfig.from_pretrained(model_name)
# DNATokenize = DNATokenizer.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, config=config)

# sequence = "AATCTA ATCTAG TCTAGC CTAGCA [SEP] TCTAGC CTAGCA"
# model_input = tokenizer.encode_plus(sequence, add_special_tokens=True, max_length=512)
# print("\n\n", model_input, "\n\n")
# model_input = bertTokenizer.encode_plus(sequence, add_special_tokens=True, max_length=512)
# print("\n\n", model_input, "\n\n")

#
# model_input = model_input["input_ids"]
# print(model_input, "\n\n")
#
# model_input = torch.tensor(model_input, dtype=torch.long)
# model_input = model_input.unsqueeze(0)  # to generate a fake batch with batch size one
# print(model_input, "\n\n")
#
# output = model(model_input)
# # print(output, "\n\n")
# print(output[0].shape, "\n\n")
# print(output[1].shape, "\n\n")

txtList = ["ATGCACGTAC" * 300 + tokenizer.sep_token + "CGTAGCATCG" * 200,
           # "ATCGGCTACT" * 300 + tokenizer.sep_token + "GCATGCATGC" * 200,
           # "TCAGAGACTG" * 300 + tokenizer.sep_token + "ACGTGCNACT" * 200,
           # "GGGTCANACT" * 300 + tokenizer.sep_token + "TTTCGAACCT" * 200
           ]
txtList1 = ("ATGCAC GTACAG TCAGAG ACTGTT ATGCGA" * 100 + tokenizer.sep_token + "CGTAGC ATCGTA GGGTCA" * 111,
            "ATCGGC TACTGT" + tokenizer.sep_token + "GCATGC ATGCGA",
            "TCAGAG ACTGTT" + tokenizer.sep_token + "ACGTGC NACTAG GGGTCA NACTAA",
            "GGGTCA NACTAA" + tokenizer.sep_token + "TTTCGA ACCTAG"
            )
txtList2 = ["CGTAGC ATCGTA",
            "GCATGC ATGCGA",
            "ACGTGC NACTAG",
            "TTTCGA ACCTAG"
            ]
txtList3 = ["ATGCAC GTACAG",
            "ATCGGC TACTGT",
            "TCAGAG ACTGTT",
            "GGGTCA NACTAA"
            ]

# encoded_inputs = tokenizer.encode_plus(txtList2[0], txtList3[0], return_tensors='pt', padding=True,
#                                        return_token_type_ids=True)
# print(encoded_inputs)
# print("***" * 48)

# right way to use dnabert with batch
encoded_inputs = tokenizer.batch_encode_plus(txtList1, return_tensors='pt', padding=True,
                                             return_token_type_ids=True, add_special_tokens=True)
print(encoded_inputs)
print("input_ids shape", encoded_inputs['input_ids'].shape)
print("***" * 48)

for item in encoded_inputs['input_ids']:
    decoder = tokenizer.decode(item)
    print(decoder)
    print(len(decoder.split()))
print("***" * 48)

feature = model(**encoded_inputs)
print(feature[0])
print(feature[0].shape)
print(type(feature[0]))
# feature = torch.as_tensor(feature)
# print(feature.shape)
print("***" * 48)
