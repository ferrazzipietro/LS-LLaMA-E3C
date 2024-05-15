from utils import DataPreprocessor, Evaluator, OutputGenerator, generate_adapters_list, generate_model_predictions
from utils import data_format_converter
from dotenv import dotenv_values

from src.billm import LlamaForTokenClassification as ModelForTokenClassification

from huggingface_hub import login
HF_TOKEN_WRITE = dotenv_values(".env.base")['HF_TOKEN_WRITE']
login(token=HF_TOKEN_WRITE)

HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']
adapters_list = generate_adapters_list('llama', appendix='5Epochs')
generate_model_predictions(adapters_list[:2])