from datetime import datetime
from .preprocessing_params import simplest_prompt


DATASET_CHEKPOINT="ferrazzipietro/e3c-sentences" 
BASE_MODEL_CHECKPOINT="meta-llama/Llama-2-7b-hf" # "meta-llama/Llama-2-7b-chat-hf"  # 
model_name=BASE_MODEL_CHECKPOINT.split('/')[1]

TRAIN_LAYER = "en.layer1"
ADAPTERS_CHECKPOINT= f"ferrazzipietro/{model_name}_adapters_{TRAIN_LAYER}"
FT_MODEL_CHECKPOINT="ferrazzipietro/ft_tmp" 
if simplest_prompt:
    ADAPTERS_CHECKPOINT=ADAPTERS_CHECKPOINT + "_simplest_prompt"

WANDB_PROJECT_NAME = f'finetune LS-{model_name}'
WANDB_RUN_NAME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

from peft import TaskType
r = [16, 32, 64] # reduce the number to finish faster
lora_alpha = [32, 64] 
lora_dropout = [0.05, 0.01]
bias =  "lora_only" 
use_rslora = True
task_type=TaskType.TOKEN_CLS
target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]# substituted by the function find_all_linear_names()

import torch

torch_dtype=torch.float16#torch.bfloat16
quantization = False
load_in_4bit=[False]
bnb_4bit_quant_type = ["nf4"]
bnb_4bit_compute_dtype = [torch.float16]#[torch.bfloat16]
llm_int8_threshold = [6.0]

bnb_4bit_use_double_quant = True
llm_int8_has_fp16_weight = True
llm_int8_skip_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]



offset=False
instruction_on_response_format='Extract the entities contained in the text. Extract only entities contained in the text.\nReturn the result in a json format: [{"entity":"entity_name"}].'
simplest_prompt=False
clent = True
