from transformers import AutoModel, AutoTokenizer, pipeline
import transformers
print(transformers.__file__)

model_name = 'pre-model/' + 'longformer-encdec-base-16384'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
classifier = pipeline('feature-extraction', model=model, tokenizer=tokenizer)

print(type(model))


