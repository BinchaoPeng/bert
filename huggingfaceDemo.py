import numpy as np
from transformers import LongformerModel, LongformerTokenizer, LongformerConfig

model_name = 'pre-model/' + 'longformer-base-4096'
config = LongformerConfig.from_pretrained(model_name)
tokenizer = LongformerTokenizer.from_pretrained(model_name)
model = LongformerModel.from_pretrained(model_name, config=config)
encoded_inputs = tokenizer("ATGCAGTA" + tokenizer.sep_token + "ATGGGCTACGT", return_tensors='pt', padding=True)
X_enpr_features = model(**encoded_inputs)
X_enpr = np.array(X_enpr_features)


