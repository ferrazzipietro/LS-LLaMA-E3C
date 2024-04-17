import json
import sys
import numpy as np
import evaluate
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from dotenv import dotenv_values
import wandb
import datetime

from utils import DataPreprocessor, DatasetFormatConverter
from modeling_llama import LlamaForTokenClassification



WANDB_KEY = dotenv_values(".env.base")['WANDB_KEY']
BASE_MODEL_CHECKPOINT = 'meta-llama/Llama-2-7b-chat-hf'
LLAMA_TOKEN = dotenv_values(".env.base")['LLAMA_TOKEN']
HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']
use_e3c = False

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_CHECKPOINT,
                                          token =LLAMA_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
# seqeval = evaluate.load("seqeval")


if not use_e3c:
    ds = load_dataset("wnut_17")
    label2id_ds = { "O": 0, "B-corporation": 1, "I-corporation": 2, "B-creative-work": 3, "I-creative-work": 4, "B-group": 5, "I-group": 6, "B-location": 7, "I-location": 8, "B-person": 9, "I-person": 10, "B-product": 11, "I-product": 12, }
    id2label_ds = {v: k for k, v in label2id_ds.items()}
    label_list_ds = list(label2id_ds.keys()) # ds["train"].features[f"ner_tags"].feature.names
    id2label = id2label_ds
    label2id = label2id_ds
    label_list = label_list_ds
    ds = ds.rename_column("ner_tags", "word_level_labels")
    ds = ds.rename_column("tokens", "words")
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
    label2id = dataset_format_converter.label2id
    id2label = {v: k for k, v in label2id.items()}
    label_list = list(label2id.keys())

def tokenize_and_align_labels(examples, max_length=1024, word_column_name='words', labels_column_name='word_level_labels'):# , word_column_name='tokens', labels_column_name='ner_tags'):#
    tokenized_inputs = tokenizer(examples[word_column_name], is_split_into_words=True, padding='longest', max_length=max_length, truncation=True)

    labels = []
    for i, label in enumerate(examples[labels_column_name]):
        # print('label: ', label)
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-99)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)# dataset_format_converter.dataset.map(tokenize_and_align_labels, batched=True)
if use_e3c:
    train_data, val_data, test_data = preprocessor.split_layer_into_train_val_test_(tokenized_ds, TRAIN_LAYER)
# tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


wandb.login(key = WANDB_KEY)
run = wandb.init(project='ls_llama_e3c', job_type="training", anonymous="allow",
                  name=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                  config={'model': BASE_MODEL_CHECKPOINT, 
                          'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

model = LlamaForTokenClassification.from_pretrained(
    BASE_MODEL_CHECKPOINT, 
    num_labels=len(label2id), 
    id2label=id2label, 
    label2id=label2id,
    token = LLAMA_TOKEN,
    load_in_4bit=True,
    device_map = 'auto',
    # cache_dir='/data/disk1/share/pferrazzi/.cache'
    )
peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS, inference_mode=False, r=12, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

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
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
    hub_token=HF_TOKEN,
    hub_model_id='ls_llama_e3c',
    report_to="wandb",
)

if use_e3c:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset= val_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
else:
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds['train'],
    eval_dataset= tokenized_ds['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    )

trainer.train()

trainer.model.push_to_hub('ls_llama_e3c',  token=HF_TOKEN) #config.ADAPTERS_CHECKPOINT, token=HF_TOKEN)
wandb.finish()