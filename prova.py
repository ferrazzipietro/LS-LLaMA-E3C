from transformers import AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
from src.billm.modeling_mistral import MistralForTokenClassification

from dotenv import dotenv_values
import torch 

HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']

# label2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
# id2label = {v: k for k, v in label2id.items()}
# model_id = 'WhereIsAI/billm-mistral-7b-conll03-ner'
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# peft_config = PeftConfig.from_pretrained(model_id)
# model = MistralForTokenClassification.from_pretrained(
#     peft_config.base_model_name_or_path,
#     num_labels=len(label2id), id2label=id2label, label2id=label2id,
#     token = HF_TOKEN,
#     # cache_dir='/data/disk1/share/pferrazzi/.cache'
# )
# model = PeftModel.from_pretrained(model, model_id)
# # merge_and_unload is necessary for inference
# model = model.merge_and_unload()

# token_classifier = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
# sentence = "I live in Hong Kong. I am a student at Hong Kong PolyU."
# tokens = token_classifier(sentence)
# print(tokens)



import json
import sys
import numpy as np
import evaluate
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from dotenv import dotenv_values
import datetime
import os
import torch



from utils import DataPreprocessor, DatasetFormatConverter
from src.billm.modeling_mistral import MistralForTokenClassification



WANDB_KEY = dotenv_values(".env.base")['WANDB_KEY']
LLAMA_TOKEN = dotenv_values(".env.base")['LLAMA_TOKEN']
HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']

adapters = "ferrazzipietro/LS_Mistral-7B-v0.1_adapters_en.layer1_NoQuant_16_32_0.01_2_0.0002"
tokenizer = AutoTokenizer.from_pretrained(adapters,
                                          token =HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
# seqeval = evaluate.load("seqeval")
peft_config = PeftConfig.from_pretrained(adapters)
BASE_MODEL_CHECKPOINT = peft_config.base_model_name_or_path
DATASET_CHEKPOINT="ferrazzipietro/e3c-sentences" 
TRAIN_LAYER="en.layer1"
offset=False
instruction_on_response_format='Extract the entities contained in the text. Extract only entities contained in the text.\nReturn the result in a json format: [{"entity":"entity_name"}].'# 'Return the result in a json format.'
simplest_prompt=False
dataset_text_field="prompt"
preprocessor = DataPreprocessor(BASE_MODEL_CHECKPOINT, 
                                tokenizer)
dataset = load_dataset(DATASET_CHEKPOINT) #download_mode="force_redownload"
dataset = dataset[TRAIN_LAYER]
dataset = dataset.shuffle(seed=1234)  # Shuffle dataset here
dataset = preprocessor.preprocess_data_one_layer(dataset, 
                                                instruction_on_response_format=instruction_on_response_format,
                                                simplest_prompt=simplest_prompt)
dataset = dataset.map(lambda samples: tokenizer(samples[dataset_text_field]), batched=True)
dataset_format_converter = DatasetFormatConverter(dataset)
dataset_format_converter.apply()
ds = dataset_format_converter.dataset
ds = ds.rename_column("word_level_labels", "ner_tags")
ds = ds.rename_column("words", "tokens")
label2id = dataset_format_converter.label2id
id2label = {v: k for k, v in label2id.items()}
label_list = list(label2id.keys())

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True, padding='longest', max_length=256, truncation=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)# dataset_format_converter.dataset.map(tokenize_and_align_labels, batched=True)
train_data, val_data, test_data = preprocessor.split_layer_into_train_val_test_(tokenized_ds, TRAIN_LAYER)
print(train_data['sentence'])

model = MistralForTokenClassification.from_pretrained(
    peft_config.base_model_name_or_path,
    num_labels=len(label2id), id2label=id2label, label2id=label2id,
    token = HF_TOKEN,
    # cache_dir='/data/disk1/share/pferrazzi/.cache'
)
model = PeftModel.from_pretrained(model, adapters)
# merge_and_unload is necessary for inference
model = model.merge_and_unload()

token_classifier = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
sentence = train_data['sentence']
tokens = token_classifier(sentence)
print(tokens)