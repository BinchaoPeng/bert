from transformers import AutoModel, AutoTokenizer, pipeline
import transformers

model_name = 'pre-model/' + 'longformer-encdec-base-16384'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
classifier = pipeline('feature-extraction', model=model, tokenizer=tokenizer)

print(type(model))

print(transformers.__file__)
