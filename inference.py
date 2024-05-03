



from dotenv import dotenv_values
from datasets import load_dataset, Dataset
from utils.data_preprocessor import DataPreprocessor
from utils.test_data_processor import TestDataProcessor
from utils.generate_ft_adapters_list import generate_ft_adapters_list
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import gc
from peft import PeftModel
from tqdm import tqdm

from src.billm.modeling_llama import LlamaForTokenClassification
from utils import DatasetFormatConverter
from config import postprocessing_params_mistral as postprocessing
from log import mistral as models_params
adapters_list = ['ferrazzipietro/LS_Mistral-7B-v0.1_adapters_en.layer1_NoQuant_16_32_0.01_2_0.0002'] #generate_ft_adapters_list("mistral_4bit", simplest_prompt=models_params.simplest_prompt)
print(adapters_list)

HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']
LLAMA_TOKEN = dotenv_values(".env.base")['LLAMA_TOKEN']

max_new_tokens_factor_list = postprocessing.max_new_tokens_factor_list
n_shots_inference_list = postprocessing.n_shots_inference_list
layer = models_params.TRAIN_LAYER
language = layer.split('.')[0]





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


DATASET_CHEKPOINT="ferrazzipietro/e3c-sentences" 
TRAIN_LAYER="en.layer1"
offset=False
instruction_on_response_format='Extract the entities contained in the text. Extract only entities contained in the text.\nReturn the result in a json format: [{"entity":"entity_name"}].'# 'Return the result in a json format.'
simplest_prompt=False
dataset_text_field="prompt"
preprocessor = DataPreprocessor(model_checkpoint=models_params.BASE_MODEL_CHECKPOINT, 
                                tokenizer = models_params.BASE_MODEL_CHECKPOINT)
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
tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)# dataset_format_converter.dataset.map(tokenize_and_align_labels, batched=True)
train_data, val_data, test_data = preprocessor.split_layer_into_train_val_test_(tokenized_ds, TRAIN_LAYER)
# Reload the base model
base_model_reload = LlamaForTokenClassification.from_pretrained(
    models_params.BASE_MODEL_CHECKPOINT, 
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
tokenizer = AutoTokenizer.from_pretrained(models_params.BASE_MODEL_CHECKPOINT, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"










# for max_new_tokens_factor in max_new_tokens_factor_list:
#     for n_shots_inference in n_shots_inference_list:
#         for adapters in tqdm(adapters_list, desc="adapters_list"):
#             print("PROCESSING:", adapters)
#             if not models_params.quantization:
#                 print("NO QUANTIZATION")
#                 base_model = AutoModelForCausalLM.from_pretrained(
#                     models_params.BASE_MODEL_CHECKPOINT, low_cpu_mem_usage=True,
#                     return_dict=True,  
#                     torch_dtype=postprocessing.torch_dtype,
#                     device_map= "auto")    
#             else:
#                 print("QUANTIZATION not implemented yet")
#                 # load_in_8bit = not models_params.load_in_4bit[0]
#                 # bnb_config = BitsAndBytesConfig(
#                 #             load_in_4bit = models_params.load_in_4bit[0],
#                 #             load_in_8bit = load_in_8bit,
#                 #             bnb_4bit_use_double_quant = models_params.bnb_4bit_use_double_quant,
#                 #             bnb_4bit_quant_type = models_params.bnb_4bit_quant_type[0],
#                 #             bnb_4bit_compute_dtype = models_params.bnb_4bit_compute_dtype[0],
#                 #             llm_int8_threshold = models_params.llm_int8_threshold[0],
#                 #             # llm_int8_has_fp16_weight = models_params.llm_int8_has_fp16_weight,
#                 #             # llm_int8_skip_modules = models_params.llm_int8_skip_modules
#                 #             )
#                 # base_model = AutoModelForCausalLM.from_pretrained(
#                 #     models_params.BASE_MODEL_CHECKPOINT, low_cpu_mem_usage=True,
#                 #     quantization_config = bnb_config,
#                 #     return_dict=True,  
#                 #     torch_dtype=torch.bfloat16,
#                 #     device_map= "auto")
#             merged_model = PeftModel.from_pretrained(base_model, adapters, token=HF_TOKEN, device_map='auto')
#             tokenizer = AutoTokenizer.from_pretrained(models_params.BASE_MODEL_CHECKPOINT, add_eos_token=True)
#             tokenizer.pad_token = tokenizer.eos_token
#             tokenizer.padding_side = "left"

#             # merged_model, tokenizer = load_mergedModel_tokenizer(adapters, base_model)
#             postprocessor = TestDataProcessor(test_data=val_data, 
#                                               preprocessor=preprocessor, 
#                                               n_shots_inference=n_shots_inference, 
#                                               language=language, 
#                                               tokenizer=tokenizer)
#             postprocessor.add_inference_prompt_column(simplest_prompt=models_params.simplest_prompt)
#             postprocessor.add_ground_truth_column()
#             # try:
#             postprocessor.add_responses_column(model=merged_model, 
#                                             tokenizer=tokenizer, 
#                                             batch_size=postprocessing.batch_size, 
#                                             max_new_tokens_factor=max_new_tokens_factor)
#             postprocessor.test_data.to_csv(f"{postprocessing.save_directory}maxNewTokensFactor{max_new_tokens_factor}_nShotsInference{n_shots_inference}_{adapters.split('/')[1]}.csv", index=False)
#             # except Exception as e:
#             #     print("ERROR IN PROCESSING: ", Exception, adapters)
#             del merged_model
#             if models_params.quantization: 
#                 del base_model
#             del tokenizer
#             gc.collect()
#             torch.cuda.empty_cache()

