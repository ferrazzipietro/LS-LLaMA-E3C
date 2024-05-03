from datetime import datetime
import torch

DATASET_CHEKPOINT="ferrazzipietro/e3c-sentences" 
BASE_MODEL_CHECKPOINT="meta-llama/Llama-2-7b-hf" # "meta-llama/Llama-2-7b-chat-hf"  # 
model_name=BASE_MODEL_CHECKPOINT.split('/')[1]

TRAIN_LAYER = "en.layer1"
ADAPTERS_CHECKPOINT= f"ferrazzipietro/{model_name}_adapters_{TRAIN_LAYER}"
FT_MODEL_CHECKPOINT="ferrazzipietro/ft_tmp" 
simplest_prompt=False

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


torch_dtype=torch.bfloat16
quantization = False
load_in_4bit=[False]
bnb_4bit_quant_type = ["nf4"]
bnb_4bit_compute_dtype = [torch.bfloat16]
llm_int8_threshold = [6.0]

bnb_4bit_use_double_quant = True
llm_int8_has_fp16_weight = True
llm_int8_skip_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]

offset=False
instruction_on_response_format='Extract the entities contained in the text. Extract only entities contained in the text.\nReturn the result in a json format: [{"entity":"entity_name"}].'

### TrainingArguments
num_train_epochs= 3
per_device_train_batch_size= 2
gradient_accumulation_steps= [2,4,8] # reduce the number to finish faster
optim = "paged_adamw_8bit"
save_steps= 1000
logging_strategy="steps"
logging_steps= 10
learning_rate= [5e-5, 2e-4]
weight_decay= 0.001
fp16= False 
bf16= True
max_grad_norm= 0.3
max_steps= -1
warmup_ratio= 0.3
group_by_length= True
lr_scheduler_type= "constant"



### SFTTrainer
"""
    max_seq_length - The maximum sequence length to use for the ConstantLengthDataset and for automatically creating the Dataset. Defaults to 512.
    dataset_text_field - The name of the field containing the text to be used for the dataset. Defaults to "text".
    packing - Used only in case dataset_text_field is passed. This argument is used by the ConstantLengthDataset to pack the sequences of the dataset.
"""
max_seq_length= 128 # It is also used to set the maximum number of tokens to be processed by the model in the definition of DatasetFormatConverter.tokenize_and_align_labels.
# longer sequences will be truncated
dataset_text_field="prompt"
packing=False
