import json
import sys
import numpy as np
import evaluate
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from dotenv import dotenv_values
import wandb
import datetime
import os
import torch

from utils import DataPreprocessor, DatasetFormatConverter
from modeling_llama import LlamaForTokenClassification



WANDB_KEY = dotenv_values(".env.base")['WANDB_KEY']
BASE_MODEL_CHECKPOINT = 'meta-llama/Llama-2-7b-chat-hf'
LLAMA_TOKEN = dotenv_values(".env.base")['LLAMA_TOKEN']
HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']
use_e3c = True


# os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_CHECKPOINT,
                                          token =LLAMA_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
seqeval = evaluate.load("seqeval")

# if not use_e3c:
#     ds = load_dataset("wnut_17")
#     label2id_ds = { "O": 0, "B-corporation": 1, "I-corporation": 2, "B-creative-work": 3, "I-creative-work": 4, "B-group": 5, "I-group": 6, "B-location": 7, "I-location": 8, "B-person": 9, "I-person": 10, "B-product": 11, "I-product": 12, }
#     id2label_ds = {v: k for k, v in label2id_ds.items()}
#     label_list_ds = list(label2id_ds.keys()) # ds["train"].features[f"ner_tags"].feature.names
#     id2label = id2label_ds
#     label2id = label2id_ds
#     label_list = label_list_ds
#     ds = ds.rename_column("ner_tags", "word_level_labels")
#     ds = ds.rename_column("tokens", "words")
if use_e3c:
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



bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,# model_loading_params.load_in_4bit,
    load_in_8bit = False,#  model_loading_params.load_in_8bit,

    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= True,

    # llm_int8_threshold= 6.0,# model_loading_params.llm_int8_threshold,
    # llm_int8_skip_modules= ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"],# model_loading_params.llm_int8_skip_modules,
    # llm_int8_has_fp16_weight= True# model_loading_params.llm_int8_has_fp16_weight
)

model = LlamaForTokenClassification.from_pretrained(
    BASE_MODEL_CHECKPOINT, 
    num_labels=len(label2id), 
    id2label=id2label, 
    label2id=label2id,
    token = LLAMA_TOKEN,
    quantization_config=bnb_config,    
    device_map = 'auto',
    # cache_dir='/data/disk1/share/pferrazzi/.cache'
    )

peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS, inference_mode=False, r=12, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)# dataset_format_converter.dataset.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


if use_e3c:
    train_data, val_data, test_data = preprocessor.split_layer_into_train_val_test_(tokenized_ds, TRAIN_LAYER)



wandb.login(key = WANDB_KEY)
run = wandb.init(project='ls_llama_e3c', job_type="training", anonymous="allow",
                  name=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                  config={'model': BASE_MODEL_CHECKPOINT, 
                          'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})




def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["soverall_f1"],
        "accuracy": results["overall_accuracy"],
    }



training_args = TrainingArguments(
    output_dir="my_awesome_ds_model",
    learning_rate=1e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps= 4,
    num_train_epochs=1,
    max_steps=10,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    #load_best_model_at_end=True,
    push_to_hub=True,
    hub_token=HF_TOKEN,
    hub_model_id='ls_llama_e3c',
    report_to="wandb",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.model.save_pretrained("ls_llama_e3c_locale") # save locally
trainer.model.push_to_hub('ls_llama_e3c',  token=HF_TOKEN) #config.ADAPTERS_CHECKPOINT, token=HF_TOKEN)
wandb.finish()

import torch
del model, trainer
torch.cuda.empty_cache()

# Reload the base model
base_model_reload = LlamaForTokenClassification.from_pretrained(
    BASE_MODEL_CHECKPOINT, 
    num_labels=len(label2id), 
    id2label=id2label, 
    label2id=label2id,
    token = LLAMA_TOKEN,
    load_in_4bit=True,
    device_map = 'auto',
    # cache_dir='/data/disk1/share/pferrazzi/.cache'
    )# .bfloat16()


merged_model = PeftModel.from_pretrained(base_model_reload, "ls_llama_e3c_locale", token=HF_TOKEN)
merged_model = merged_model.merge_and_unload()


# Reload tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_CHECKPOINT, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# Push the model and tokenizer to the Hugging Face Model Hub
merged_model.push_to_hub("ferrazzipietro/ls_llama_e3c_model", use_temp_dir=False, token=HF_TOKEN )
tokenizer.push_to_hub("ferrazzipietro/ls_llama_e3c_model", use_temp_dir=False, token=HF_TOKEN )