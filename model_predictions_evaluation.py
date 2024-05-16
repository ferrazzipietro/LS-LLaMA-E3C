from huggingface_hub import login
from datasets import load_dataset
from dotenv import dotenv_values
import os
import torch
import pandas as pd
from peft import PeftConfig
from transformers import AutoTokenizer
from utils import Evaluator, generate_adapters_list

HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']
HF_TOKEN_WRITE = dotenv_values(".env.base")['HF_TOKEN_WRITE']
login(token=HF_TOKEN_WRITE)


appendix = '5Epochs' # 5EpochsBestF1Train
model_type ='llama'

def extract_params_from_file_name(df: pd.DataFrame):
    df['model_type'] = df['dataset'].apply(lambda x: str(x.split('/')[1].split('_')[1]))
    df['training_config'] = df['dataset'].apply(lambda x: str(x.split('adapters_')[1]))
    df['layer'] = df['dataset'].apply(lambda x: str(x.split('/')[1].split('_')[3]))
    df['quantization'] = df['dataset'].apply(lambda x: str(x.split('/')[1].split('_')[4]))
    df['quantization'] = df['dataset'].apply(lambda x: str(x.split('/')[1].split('_')[4]))
    df['r'] = df['dataset'].apply(lambda x: str(x.split('/')[1].split('_')[5]))
    df['lora_alpha'] = df['dataset'].apply(lambda x: str(x.split('/')[1].split('_')[6]))
    df['lora_dropout'] = df['dataset'].apply(lambda x: str(x.split('/')[1].split('_')[7]))
    df['gradient_accumulation_steps'] = df['dataset'].apply(lambda x: str(x.split('/')[1].split('_')[8]))
    df['learning_rate'] = df['dataset'].apply(lambda x: str(x.split('/')[1].split('_')[9]))
    df['run_type'] = df['dataset'].apply(lambda x: str(x.split('/')[1].split('_')[10]))
    return df

datasets_list = generate_adapters_list(model_type, appendix=appendix)
peft_config = PeftConfig.from_pretrained(datasets_list[0], token = HF_TOKEN_WRITE)
BASE_MODEL_CHECKPOINT = peft_config.base_model_name_or_path
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_CHECKPOINT,token =HF_TOKEN_WRITE)

evaluation_table = pd.DataFrame(columns=['dataset', 'TP', 'FP', 'FN', 'precision', 'recall', 'f1'])

for i, dataset_checkpoint in enumerate(datasets_list):
    print(f"evaluating {dataset_checkpoint}, {i}/{len(datasets_list)}...")
    test_data = load_dataset(dataset_checkpoint, token=HF_TOKEN, split='test')
    eval = Evaluator(test_data, tokenizer)
    eval.extract_FP_FN_TP_token_by_token()
    eval.create_evaluation_table()
    tmp = eval.evaluation_table.copy()
    tmp['dataset'] = dataset_checkpoint
    evaluation_table = pd.concat([evaluation_table, pd.DataFrame([tmp])])#  evaluation_table.con(tmp)
    print(eval.evaluation_table)

print(evaluation_table)
evaluation_table#.to_csv(f'data/evaluation_table{appendix}.csv', index=False)
evaluation_table = extract_params_from_file_name(evaluation_table)
evaluation_table.to_csv(f'data/evaluation_table{appendix}.csv', index=False)
print(f'SAVED TO data/evaluation_table{appendix}.csv')