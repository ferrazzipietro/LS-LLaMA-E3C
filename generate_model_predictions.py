from utils import generate_adapters_list, generate_model_predictions, OutputGenerator, DataPreprocessor
from utils.data_format_converter import  DatasetFormatConverter
from dotenv import dotenv_values
import os 
import torch


from huggingface_hub import login



from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from dotenv import dotenv_values
import torch
from tqdm.auto import tqdm
import gc



from src.billm import LlamaForTokenClassification, MistralForTokenClassification



def generate_model_predictions(adapters_list: 'list[str]', batch_size = 32):
    DATASET_CHEKPOINT="ferrazzipietro/e3c-sentences" 
    TRAIN_LAYER="en.layer1"
    peft_config = PeftConfig.from_pretrained(adapters_list[0], token = HF_TOKEN_WRITE)
    BASE_MODEL_CHECKPOINT = peft_config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_CHECKPOINT,token =HF_TOKEN_WRITE)
    tokenizer.pad_token = tokenizer.eos_token
    print('PREPROCESSING DATA...')
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
    preprocessor = DataPreprocessor()
    _, data, _ = preprocessor.split_layer_into_train_val_test_(tokenized_ds, TRAIN_LAYER)
    print('PREPROCESSING DATA...DONE')
    
    for adapters in adapters_list:
        model_type = 'llama' if 'llama' in adapters.lower() else 'mistral'
        print('loading model with adapters:', adapters)
        if model_type == 'llama':
            ModelForTokenClassification = LlamaForTokenClassification
        elif model_type == 'mistral':
            ModelForTokenClassification = MistralForTokenClassification
        else:
            raise ValueError('Model type not recognized')
        peft_config = PeftConfig.from_pretrained(adapters, token = HF_TOKEN_WRITE)
        BASE_MODEL_CHECKPOINT = peft_config.base_model_name_or_path
        model = ModelForTokenClassification.from_pretrained(
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
        test_data = generator.generate(data.select(range(4)), batch_size = batch_size)
        print(test_data)
        test_data.push_to_hub(adapters, token=HF_TOKEN_WRITE, split='test')
        del model
        gc.collect()
        torch.cuda.empty_cache()

HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']
HF_TOKEN_WRITE = dotenv_values(".env.base")['HF_TOKEN_WRITE']
login(token=HF_TOKEN_WRITE)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.cuda.empty_cache() 
adapters_list = generate_adapters_list('llama', appendix='5Epochs')
generate_model_predictions(adapters_list, batch_size=2)