import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from peft import PeftModel, PeftConfig
from dotenv import dotenv_values
import torch
from tqdm.auto import tqdm
import evaluate

from utils import DataPreprocessor, Evaluator, OutputGenerator
from utils import data_format_converter

from src.billm import LlamaForTokenClassification

WANDB_KEY = dotenv_values(".env.base")['WANDB_KEY']
LLAMA_TOKEN = dotenv_values(".env.base")['LLAMA_TOKEN']
HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']
HF_TOKEN_WRITE = dotenv_values(".env.base")['HF_TOKEN_WRITE']


adapters = "ferrazzipietro/noLoraLS_Llama-2-7b-hf_adapters_en.layer1_NoQuant_2_0.0002_5EpochsBestF1Train"# "ferrazzipietro/LS_Llama-2-7b-hf_adapters_en.layer1_NoQuant_64_32_0.01_2_0.0002_5Epochs"
peft_config = PeftConfig.from_pretrained(adapters, token = HF_TOKEN)
BASE_MODEL_CHECKPOINT = peft_config.base_model_name_or_path

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_CHECKPOINT,token =HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
# seqeval = evaluate.load("seqeval")
DATASET_CHEKPOINT="ferrazzipietro/e3c-sentences" 
TRAIN_LAYER="en.layer1"
preprocessor = DataPreprocessor()
dataset = load_dataset(DATASET_CHEKPOINT) #download_mode="force_redownload"
dataset = dataset[TRAIN_LAYER]
dataset = dataset.shuffle(seed=1234)  
dataset_format_converter = data_format_converter.DatasetFormatConverter(dataset)
dataset_format_converter.apply()
ds = dataset_format_converter.dataset
label2id = dataset_format_converter.label2id
id2label = dataset_format_converter.get_id2label()
label_list = dataset_format_converter.get_label_list()
dataset_format_converter.set_tokenizer(tokenizer)
dataset_format_converter.set_max_seq_length(256)
tokenized_ds = ds.map(lambda x: dataset_format_converter.tokenize_and_align_labels(x), batched=True)# dataset_format_converter.dataset.map(tokenize_and_align_labels, batched=True)
_, val_data, _ = preprocessor.split_layer_into_train_val_test_(tokenized_ds, TRAIN_LAYER)


model = LlamaForTokenClassification.from_pretrained(
    peft_config.base_model_name_or_path,
    num_labels=len(label2id), id2label=id2label, label2id=label2id,
    token = HF_TOKEN,
    # cache_dir='/data/disk1/share/pferrazzi/.cache',
    device_map='auto')
model = PeftModel.from_pretrained(model, adapters, token = HF_TOKEN)
model = model.merge_and_unload()

generator = OutputGenerator(model, tokenizer, label2id, label_list)
test_data = generator.generate(val_data, batch_size = 84)
test_data.push_to_hub(adapters, token=HF_TOKEN_WRITE)