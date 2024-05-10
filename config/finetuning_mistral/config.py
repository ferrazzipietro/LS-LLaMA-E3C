from datetime import datetime

DATASET_CHEKPOINT="ferrazzipietro/e3c-sentences" 
BASE_MODEL_CHECKPOINT="mistralai/Mistral-7B-v0.1" #  "mistralai/Mistral-7B-v0.1"  # "mistralai/Mistral-7B-v0.1" # "mistralai/Mistral-7B-Instruct-v0.2"
model_name=BASE_MODEL_CHECKPOINT.split('/')[1]

TRAIN_LAYER = "en.layer1"

WANDB_PROJECT_NAME = f'finetune LS-{model_name}'

