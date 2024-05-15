from utils import generate_adapters_list, generate_model_predictions
from dotenv import dotenv_values
import os 
import torch


from huggingface_hub import login

HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']
HF_TOKEN_WRITE = dotenv_values(".env.base")['HF_TOKEN_WRITE']
login(token=HF_TOKEN_WRITE)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:<enter-size-here>"
torch.cuda.empty_cache() 
adapters_list = generate_adapters_list('llama', appendix='5Epochs')
generate_model_predictions(adapters_list, batch_size=2)