# https://colab.research.google.com/github/adithya-s-k/LLM-Alchemy-Chamber/blob/main/LLMs/Mistral-7b/Mistral_Colab_Finetune_ipynb_Colab_Final.ipynb?source=post_page-----0f39647b20fe--------------------------------#scrollTo=acCr5AZ0831z

import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForTokenClassification, TrainingArguments, Trainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, TaskType
from trl import SFTTrainer
import wandb
import datetime
import gc
import evaluate
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from dotenv import dotenv_values
import torch
from tqdm.auto import tqdm

from utils.data_preprocessor import DataPreprocessor
from utils import data_format_converter
# import importlib
# importlib.reload(dataset_format_converter)

#from src.billm import MistralForTokenClassification
from src.billm import LlamaForTokenClassification


from config.finetuning_llama2 import training_params, lora_params, model_loading_params, config, preprocessing_params

seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    # print('predictions:\n',predictions)
    # print('size pred:',predictions.shape)

    # print('labels:\n',labels)
    # print('labels pred:',labels.shape)
    predictions = np.argmax(predictions, axis=2)
    #print('predictions:\n',predictions)
    #print('len(predictions[0]):\n',len(predictions[0]))
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    # print('true_predictions: ', true_predictions)
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    # print('true_labels: ', true_labels)

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    # print(results)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

WANDB_KEY = dotenv_values(".env.base")['WANDB_KEY']
LLAMA_TOKEN = dotenv_values(".env.base")['LLAMA_TOKEN']
HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']
HF_TOKEN_WRITE = dotenv_values(".env.base")['HF_TOKEN_WRITE']


def main(ADAPTERS_CHECKPOINT,
         #load_in_4bit, bnb_4bit_quant_type, bnb_4bit_compute_dtype, llm_int8_threshold, llm_int8_skip_modules,
         r, lora_alpha, lora_dropout,
         gradient_accumulation_steps,learning_rate,
         tokenizer):
    
    # Monitering the LLM
    wandb.login(key = WANDB_KEY)
    run = wandb.init(project=config.WANDB_PROJECT_NAME, job_type="training", anonymous="allow",
                    name=ADAPTERS_CHECKPOINT.split('/')[1],
                    config={'model': config.BASE_MODEL_CHECKPOINT, 
                            'dataset': config.DATASET_CHEKPOINT, 
                            'layer': config.TRAIN_LAYER,
                            'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    if not model_loading_params.quantization:
        model = LlamaForTokenClassification.from_pretrained(
        config.BASE_MODEL_CHECKPOINT,
        num_labels=len(label2id), 
        id2label=id2label, 
        label2id=label2id,
        device_map="auto",
        token=HF_TOKEN,
        torch_dtype=model_loading_params.torch_dtype,
        #cache_dir='/data/disk1/share/pferrazzi/.cache'
        )
        model.gradient_checkpointing_enable() # Activates gradient checkpointing for the current model.
        model.config.use_cache = False
    else:
        raise ValueError("Quantization is not supported by TokenClassification models. Please remove the quantization flag from the model_loading_params")
        # bnb_config = BitsAndBytesConfig(
        # load_in_4bit= load_in_4bit,
        # load_in_8bit = load_in_8bit,

        # bnb_4bit_quant_type= bnb_4bit_quant_type,
        # bnb_4bit_compute_dtype= bnb_4bit_compute_dtype,
        # bnb_4bit_use_double_quant= model_loading_params.bnb_4bit_use_double_quant,

        # llm_int8_threshold= llm_int8_threshold,
        # llm_int8_skip_modules= llm_int8_skip_modules,
        # # llm_int8_has_fp16_weight= model_loading_params.llm_int8_has_fp16_weight # Had to comment this to run llama 7B in 8 bit. There are numerical issues with fp16. I will instead use the default float16
        # )

        # model = MistralForTokenClassification.from_pretrained(
        #     config.BASE_MODEL_CHECKPOINT,
        #     quantization_config=bnb_config,
        #     device_map="auto",
        #     token=LLAMA_TOKEN
        # )
        # """
        # prepare_model_for_kbit_training wraps the entire protocol for preparing a model before running a training. 
        #         This includes:  1- Cast the layernorm in fp32 
        #                         2- making output embedding layer require gradient (needed as you are going to train (finetune) the model)
        #                         3- upcasting the model's head to fp32 for numerical stability
        # """
        # model = prepare_model_for_kbit_training(model)
    #model.gradient_checkpointing_enable() # Activates gradient checkpointing for the current model.
    #model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    lora_config = LoraConfig(
            task_type=lora_params.task_type,
            inference_mode=False, 
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            # bias=lora_params.bias,
            #  target_modules=lora_params.target_modules # lora_params.target_modules
            )
    model = get_peft_model(model, lora_config)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    train_data, val_data, test_data = preprocessor.split_layer_into_train_val_test_(tokenized_ds, config.TRAIN_LAYER)
    torch.cuda.empty_cache()

    #Hyperparamter
    training_arguments = TrainingArguments(
        output_dir= "./training_output",
        learning_rate= learning_rate,
        per_device_train_batch_size= training_params.per_device_train_batch_size,
        per_device_eval_batch_size= training_params.per_device_train_batch_size,
        num_train_epochs= training_params.num_train_epochs,
        weight_decay= training_params.weight_decay,
        hub_token=HF_TOKEN,
        hub_private_repo=True,
        push_to_hub=True,
        hub_model_id=ADAPTERS_CHECKPOINT,
        evaluation_strategy = training_params.evaluation_strategy,
        save_strategy = training_params.save_strategy,
        eval_steps = training_params.eval_steps,
        greater_is_better = training_params.greater_is_better,
        metric_for_best_model = training_params.metric_for_best_model,
        save_total_limit = training_params.save_total_limit,
        load_best_model_at_end = training_params.load_best_model_at_end,
        gradient_accumulation_steps= gradient_accumulation_steps,
        optim=  training_params.optim,
        save_steps= training_params.save_steps,
        logging_strategy=training_params.logging_strategy,
        logging_steps= training_params.logging_steps,
        fp16= training_params.fp16,
        bf16= training_params.bf16,
        max_grad_norm= training_params.max_grad_norm,
        max_steps= training_params.max_steps,
        warmup_ratio= training_params.warmup_ratio,
        group_by_length= training_params.group_by_length,
        lr_scheduler_type= training_params.lr_scheduler_type,
        report_to="wandb",
        #lr_scheduler_type="cosine",
        #warmup_ratio = 0.1,
        # logging strategies 
        # remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_data,
        eval_dataset=val_data,
        # dataset_text_field=training_params.dataset_text_field,
        # peft_config=lora_config,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        # max_seq_length = training_params.max_seq_length
    )

    trainer.train()
    results = trainer.evaluate()

    # Print evaluation results
    print(f"The results on the final model are: {results}")

    # trainer.model.save_pretrained(f"{config.BASE_MODEL_CHECKPOINT.split('/')[1]}_prova") # save locally
    trainer.model.push_to_hub(ADAPTERS_CHECKPOINT, token=HF_TOKEN_WRITE, split='test')

    wandb.finish()
    del model
    del trainer
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()



if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_CHECKPOINT,
                                            token = HF_TOKEN) #, cache_dir='/data/disk1/share/pferrazzi/.cache')
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = 'right'
    # seqeval = evaluate.load("seqeval")


    preprocessor = DataPreprocessor(config.BASE_MODEL_CHECKPOINT, 
                                        tokenizer)
    dataset = load_dataset(config.DATASET_CHEKPOINT) #download_mode="force_redownload"
    dataset = dataset[config.TRAIN_LAYER]
    dataset = dataset.shuffle(seed=1234)  # Shuffle dataset here
    dataset_format_converter_obj = data_format_converter.DatasetFormatConverter(dataset)
    dataset_format_converter_obj.apply()
    ds = dataset_format_converter_obj.dataset
    label2id = dataset_format_converter_obj.label2id
    id2label = dataset_format_converter_obj.get_id2label()
    label_list = dataset_format_converter_obj.get_label_list()
    dataset_format_converter_obj.set_tokenizer(tokenizer)
    dataset_format_converter_obj.set_max_seq_length(training_params.max_seq_length)
    tokenized_ds = ds.map(lambda x: dataset_format_converter_obj.tokenize_and_align_labels(x), batched=True)# 
    train_data, val_data, test_data = preprocessor.split_layer_into_train_val_test_(tokenized_ds, config.TRAIN_LAYER)
    print(train_data[0]['labels'])

    # load_in_4bit_list = model_loading_params.load_in_4bit
    # bnb_4bit_quant_type_list = model_loading_params.bnb_4bit_quant_type
    # bnb_4bit_compute_dtype_list = model_loading_params.bnb_4bit_compute_dtype
    # llm_int8_threshold_list = model_loading_params.llm_int8_threshold
    r_list = lora_params.r
    lora_alpha_list = lora_params.lora_alpha
    lora_dropout_list = lora_params.lora_dropout
    gradient_accumulation_steps_list = training_params.gradient_accumulation_steps
    learning_rate_list = training_params.learning_rate
    
    # load_in_4bit = load_in_4bit_list[model_loading_params_idx]
    # load_in_8bit = not load_in_4bit
    # bnb_4bit_quant_type = bnb_4bit_quant_type_list[model_loading_params_idx]
    # bnb_4bit_compute_dtype = bnb_4bit_compute_dtype_list[model_loading_params_idx]
    # llm_int8_threshold = llm_int8_threshold_list[model_loading_params_idx]
    # print('I AM LOADING A MODEL IN load_in_4bit=', load_in_4bit, 'load_in_8bit=', load_in_8bit, 'bnb_4bit_quant_type=', bnb_4bit_quant_type, 'bnb_4bit_compute_dtype=', bnb_4bit_compute_dtype, 'llm_int8_threshold=', llm_int8_threshold)
    for r in r_list:
        for lora_alpha in lora_alpha_list:
            for lora_dropout in lora_dropout_list:
                for gradient_accumulation_steps in gradient_accumulation_steps_list:
                    for learning_rate in learning_rate_list:
                        # nbits = 4
                        # if load_in_8bit:
                        #     nbits = 8
                        if not model_loading_params.quantization:
                            nbits = "NoQuant"
                            extra_str = ""
                        if preprocessing_params.simplest_prompt:
                            extra_str = "simplest_prompt_"
                        else:
                            extra_str = ""
                        ADAPTERS_CHECKPOINT = f"ferrazzipietro/LS_{config.model_name}_{extra_str}adapters_{config.TRAIN_LAYER}_{nbits}_{r}_{lora_alpha}_{lora_dropout}_{gradient_accumulation_steps}_{learning_rate}_5Epochs"
                        main(ADAPTERS_CHECKPOINT,
                            # load_in_4bit, bnb_4bit_quant_type, bnb_4bit_compute_dtype, llm_int8_threshold,
                            r, lora_alpha, lora_dropout,
                            gradient_accumulation_steps,learning_rate,
                            tokenizer)
                        gc.collect()
                        torch.cuda.empty_cache()
            