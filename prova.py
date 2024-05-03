from transformers import AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
from src.billm.modeling_mistral import MistralForTokenClassification

from dotenv import dotenv_values
import torch 

HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']

label2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
id2label = {v: k for k, v in label2id.items()}
model_id = 'WhereIsAI/billm-mistral-7b-conll03-ner'
tokenizer = AutoTokenizer.from_pretrained(model_id)
peft_config = PeftConfig.from_pretrained(model_id)
model = MistralForTokenClassification.from_pretrained(
    peft_config.base_model_name_or_path,
    num_labels=len(label2id), id2label=id2label, label2id=label2id,
    token = HF_TOKEN,
    cache_dir='/data/disk1/share/pferrazzi/.cache'
)
model = PeftModel.from_pretrained(model, model_id)
# merge_and_unload is necessary for inference
model = model.merge_and_unload()

token_classifier = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
sentence = "I live in Hong Kong. I am a student at Hong Kong PolyU."
tokens = token_classifier(sentence)
print(tokens)
