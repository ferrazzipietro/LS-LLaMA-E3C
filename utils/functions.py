import importlib
from dotenv import dotenv_values

HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']
HF_TOKEN_WRITE = dotenv_values(".env.base")['HF_TOKEN_WRITE']

def generate_adapters_list(log_run_name: str, appendix:str) -> 'list[str]':
    """
    Given a run of several configurations for the qlora fine-tuning, this function returns the list of the adapters generated by the run.
    """
    adapters_list = []
    module_name = f"log.{log_run_name}"
    models_params = importlib.import_module(module_name)
    
    model_name = models_params.model_name
    quantization = models_params.quantization
    load_in_4bit_list = models_params.load_in_4bit
    bnb_4bit_quant_type_list = models_params.bnb_4bit_quant_type
    bnb_4bit_compute_dtype_list = models_params.bnb_4bit_compute_dtype
    llm_int8_threshold_list = models_params.llm_int8_threshold
    r_list = models_params.r
    lora_alpha_list = models_params.lora_alpha
    lora_dropout_list = models_params.lora_dropout
    gradient_accumulation_steps_list = models_params.gradient_accumulation_steps
    learning_rate_list = models_params.learning_rate
    
    for model_loading_params_idx in range(len(load_in_4bit_list)):
        load_in_4bit = load_in_4bit_list[model_loading_params_idx]
        load_in_8bit = not load_in_4bit
        bnb_4bit_quant_type = bnb_4bit_quant_type_list[model_loading_params_idx]
        bnb_4bit_compute_dtype = bnb_4bit_compute_dtype_list[model_loading_params_idx]
        llm_int8_threshold = llm_int8_threshold_list[model_loading_params_idx]
        for r in r_list:
            for lora_alpha in lora_alpha_list:
                for lora_dropout in lora_dropout_list:
                    for gradient_accumulation_steps in gradient_accumulation_steps_list:
                        for learning_rate in learning_rate_list:
                            nbits = 4
                            if load_in_8bit:
                                nbits = 8   
                            if not quantization:
                                nbits = 'NoQuant'
                            ADAPTERS_CHECKPOINT = f"ferrazzipietro/LS_{model_name}_adapters_{models_params.TRAIN_LAYER}_{nbits}_{r}_{lora_alpha}_{lora_dropout}_{gradient_accumulation_steps}_{learning_rate}_{appendix}"
                            adapters_list.append(ADAPTERS_CHECKPOINT)
    return adapters_list


from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from dotenv import dotenv_values
import torch
from tqdm.auto import tqdm
import gc


from .output_generator import OutputGenerator
from .data_format_converter import DatasetFormatConverter
from .data_preprocessor import DataPreprocessor

from src.billm import LlamaForTokenClassification, MistralForTokenClassification



def generate_model_predictions(adapters_list: 'list[str]', batch_size = 32):
    DATASET_CHEKPOINT="ferrazzipietro/e3c-sentences" 
    TRAIN_LAYER="en.layer1"
    preprocessor = DataPreprocessor(BASE_MODEL_CHECKPOINT, 
                                    tokenizer)
    dataset = load_dataset(DATASET_CHEKPOINT, token=HF_TOKEN_WRITE) #download_mode="force_redownload"
    dataset = dataset[TRAIN_LAYER]
    dataset = dataset.shuffle(seed=1234)  
    dataset_format_converter = DatasetFormatConverter(dataset)
    dataset_format_converter.apply()
    ds = dataset_format_converter.dataset
    label2id = dataset_format_converter.label2id
    id2label = dataset_format_converter.get_id2label()
    label_list = dataset_format_converter.get_label_list()
    dataset_format_converter.set_tokenizer(tokenizer)
    dataset_format_converter.set_max_seq_length(256)
    tokenized_ds = ds.map(lambda x: dataset_format_converter.tokenize_and_align_labels(x), batched=True)
    _, data, _ = preprocessor.split_layer_into_train_val_test_(tokenized_ds, TRAIN_LAYER)
    
    for adapters in adapters_list:
        model_type = 'llama' if 'llama' in adapters.lower() else 'mistral'
        print('preprocessing data and loading model with adapters:', adapters)
        if model_type == 'llama':
            ModelForTokenClassification = LlamaForTokenClassification
        elif model_type == 'mistral':
            ModelForTokenClassification = MistralForTokenClassification
        else:
            raise ValueError('Model type not recognized')
        peft_config = PeftConfig.from_pretrained(adapters, token = HF_TOKEN_WRITE)
        BASE_MODEL_CHECKPOINT = peft_config.base_model_name_or_path

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_CHECKPOINT,token =HF_TOKEN_WRITE)
        tokenizer.pad_token = tokenizer.eos_token
        base_model = ModelForTokenClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            num_labels=len(label2id), id2label=id2label, label2id=label2id,
            token = HF_TOKEN_WRITE,
            # cache_dir='/data/disk1/share/pferrazzi/.cache',
            device_map='auto',
            # quantization_config = bnb_config
            )
        model = PeftModel.from_pretrained(model, adapters, token = HF_TOKEN_WRITE)
        model = model.merge_and_unload()
        print('DONE')
        generator = OutputGenerator(model, tokenizer, label2id, label_list)
        test_data = generator.generate(data, batch_size = batch_size)
        print(test_data)
        test_data.push_to_hub(adapters, token=HF_TOKEN_WRITE, 
                              split='test' )
        del model
        gc.collect()
        torch.cuda.empty_cache()