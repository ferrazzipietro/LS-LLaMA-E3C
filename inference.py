
import json
import sys
import numpy as np
import evaluate
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from dotenv import dotenv_values
import datetime
import os


from utils import DataPreprocessor, DatasetFormatConverter
from billm.modeling_mistralbillm import MistralForTokenClassification



WANDB_KEY = dotenv_values(".env.base")['WANDB_KEY']
BASE_MODEL_CHECKPOINT = 'mistralai/Mistral-7B-Instruct-v0.2'
LLAMA_TOKEN = dotenv_values(".env.base")['LLAMA_TOKEN']
HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_CHECKPOINT,
                                          token =HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
# seqeval = evaluate.load("seqeval")

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


base_model = MistralForTokenClassification.from_pretrained(
    BASE_MODEL_CHECKPOINT, 
    num_labels=len(label2id), 
    id2label=id2label, 
    label2id=label2id,
    token = LLAMA_TOKEN,
    device_map = 'auto',
    cache_dir='/data/disk1/share/pferrazzi/.cache'
    ).bfloat16()
adapters = "ferrazzipietro/LS_Mistral-7B-v0.1_adapters_en.layer1_NoQuant_16_32_0.01_2_0.0002"
merged_model = PeftModel.from_pretrained(base_model, 
                                                     adapters, 
                                                     token=HF_TOKEN, 
                                                     device_map='auto',
                                                     is_trainable = False)


import torch

class Postprocessor():
    def __init__(self, tokenizer, label_list):
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.id2label = {v: k for k, v in label_list.items()}
        self.label2id = label_list
    def postprocess(self, model_output):
        model_output_logits = model_output.logits.cpu().detach().numpy()
        preds = np.argmax(model_output_logits, axis=2)
        preds_list = []
        for pred in preds:
            preds_list.append([self.id2label[label] for label in pred])
        return preds_list


device = "cuda"
tokenizer.padding_side = "left"
# examples=['The doctor went to the hospital Precipitevolissimevolmente', 'the 12 years old girl went to see the doctor']
lista = [4 * i for i in range(1, 7)]

for i in range(len(lista)-1):
    examples = train_data['sentence'][lista[i]:lista[i+1]]
    input_sentences = examples
    encodeds = tokenizer(input_sentences, return_tensors="pt", add_special_tokens=False, padding=True)
    model_inputs = encodeds.to(device)
    generated_ids = merged_model(**model_inputs,)
    postprocessor = Postprocessor(tokenizer, label2id)
    preds_list = postprocessor.postprocess(generated_ids)

    for el in range(len(examples)):
        tokens = tokenizer.convert_ids_to_tokens(encodeds['input_ids'][el])
        for k in range(len(tokens)):
            print(f"{tokens[k]} : {preds_list[el][k]}")
