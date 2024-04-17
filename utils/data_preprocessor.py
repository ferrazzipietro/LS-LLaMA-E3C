from datasets import Dataset
import os
import random
from transformers import AutoTokenizer
import warnings

class DataPreprocessor():


    def __init__(self, model_checkpoint:str, tokenizer: AutoTokenizer, token_llama:str='') -> None:

        self.offset = None
        self.instruction_on_response_format = ''
        self.n_shots = None
        #self.model_type = model_checkpoint.split('/')[1].lower().split('-')[0]
        self.model_type = 'qwen' if model_checkpoint.split('/')[0] == 'Qwen' else model_checkpoint.split('/')[1].lower().split('-')[0]
        # if self.model_type == 'zefiro':
        #     self.model_type  = 'mistral'
        if self.model_type not in ['mistral', 'llama', 'gemma', 'qwen', 'zefiro']:
            raise ValueError("The model type must be either 'mistral', 'llama', 'gemma', 'zefiro' or 'qwen'")

        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, token = token_llama)
        else:
            self.tokenizer = tokenizer
        
        self.special_tokens_instruction_dict = {'mistral': {'user_start':'[INST]',
                                                            'user_end':'[/INST]',
                                                            'model_start':'',
                                                            'model_end':''},
                                                'llama': {'user_start':'[INST]',
                                                          'user_end':'[/INST]',
                                                          'model_start':'',
                                                          'model_end':''},
                                                'gemma': {'user_start':'<start_of_turn>user',
                                                          'user_end':'<end_of_turn>',
                                                          'model_start':'<start_of_turn>model',
                                                          'model_end':'<end_of_turn>'},
                                                'qwen': {'user_start':'<|im_start|>user',
                                                          'user_end':'<|im_end|>',
                                                          'model_start':'<|im_start|>assistant',
                                                          'model_end':'<|im_end|>'},
                                                'zefiro': {'user_start':'<|user|>',
                                                           'user_end':'</s>',
                                                           'model_start':'<|assistant|>',
                                                           'model_end':'</s>'}}
        self.special_tokens_instruction = self.special_tokens_instruction_dict[self.model_type]

        self.one_shot_example = """{user_start} {instruction_on_response_format} <<<{example_query}>>> {user_end}{model_start} {example_response} {model_end}
"""
        self.one_shot_example_no_offset = """{user_start} {instruction_on_response_format} <<<{example_query}>>> {user_end}{model_start} {example_response} {model_end}
"""

        self.prompt_template = """{user_start} {instruction_on_response_format} <<{query}>>> {user_end}{model_start}"""

        self.prompt_template_no_offset = """{user_start} {instruction_on_response_format} <<{query}>>> {user_end}{model_start}"""

    def _base_prompt_input(self, input: str, instruction_on_response_format:str) -> str:
        """
        Format the input into a base prompt for the finetuning

        Args:
            input: the input text
            instruction_on_response_format: the instruction on the response format. E.g. "The response must be a list of dictionaries, where each dictionary contains the keys 'text' and 'offset'"

        Returns:
            the formatted base prompt
        """
        base_prompt = self.prompt_template_no_offset.format(
            instruction_on_response_format=instruction_on_response_format, 
            query=input,
            user_start=self.special_tokens_instruction['user_start'],
            user_end=self.special_tokens_instruction['user_end'],
            model_start=self.special_tokens_instruction['model_start'],
            model_end=self.special_tokens_instruction['model_end'])
            
        return base_prompt

    def _simplest_base_prompt_input(self, input: str) -> str:
        """
        Format the input and output into a prompt for the finetuning, in the simplest way possible, containing only the sentence and the response

        Args:
            input: the input text
            output: the output text

        Returns:
            the formatted prompt
        """
        base_prompt = self.special_tokens_instruction['user_start'] + input + self.special_tokens_instruction['user_end'] + self.special_tokens_instruction['model_start']
        return base_prompt

    def _format_prompt(self, input: str, instruction_on_response_format:str, simplest_prompt: bool, output:str='') -> str:
        """
        Format the input and output into a prompt for the finetuning

        Args:
            input: the input text
            instruction_on_response_format: the instruction on the response format. E.g. "The response must be a list of dictionaries, where each dictionary contains the keys 'text' and 'offset'"
            offset: whether to require the offset in the response
            output: the output text

        Returns:
            the formatted prompt
        """
        if output == '':
            raise ValueError("The output must be provided when generating prompts for the finetuning")
        
        if simplest_prompt:
            prompt_input = self._simplest_base_prompt_input(input)
        else:
            prompt_input = self._base_prompt_input(input, instruction_on_response_format)
        
        bos_token = self.tokenizer.bos_token
        eos_token = self.tokenizer.eos_token
        if self.model_type == 'qwen':
            bos_token = ''
            eos_token = ''
        # print(bos_token, prompt_input, output, self.special_tokens_instruction['model_end'], eos_token)
        prompt = bos_token + prompt_input + output + self.special_tokens_instruction['model_end'] + eos_token
                            
        return prompt


    def _format_entities_in_response(self, entities_list: [dict], offset: bool) -> str:
        """
        Format the response into a string

        Args:
            entities_list: the list of entities to format
            offset: whether to require the offset in the response
            
        Returns:
            the formatted response
        """
        formatted_response = '['
        if offset:
            for entity in entities_list:
                formatted_response = formatted_response + '{"entity": "' + entity['text'] + f'", "offset": {entity["offsets"]}' + '}, '
        else:
            for entity in entities_list: 
                formatted_response = formatted_response + '{"entity": "' + entity['text'] + '"}, '
        formatted_response = formatted_response[:-2]
        formatted_response = formatted_response + '] '
        return formatted_response
    
    def _apply_to_one_example(self, example, offset: bool, simplest_prompt: bool, instruction_on_response_format:str) -> dict:
        """
        Apply the data preprocessing to one example

        Args:
            example: the example (data row) to preprocess
            instruction_on_response_format: the instruction on the response format. E.g. "The response must be a list of dictionaries, where each dictionary contains the keys 'text' and 'offset'"
            offset: whether to require the offset in the response
            simplest_prompt: whether to generate the prompt or just concatenate the sentence and the response

        Returns:
            the preprocessed example
        """
        output = self._format_entities_in_response(entities_list=example['entities'], offset=offset)
        prompt = self._format_prompt(input=example['sentence'], 
                                     simplest_prompt=simplest_prompt,
                                     instruction_on_response_format=instruction_on_response_format,
                                     output=output)
        example['prompt'] = prompt
        return example
    
    def apply(self, data: Dataset, instruction_on_response_format:str, offset: bool,  simplest_prompt:bool, num_proc: int=1) -> Dataset:
        """
        Apply the data preprocessing to one split/layer if the dataset. It formats the prompt in the right shape, processing the entities.

        Args:
            data: the dataset to preprocess
            instruction_on_response_format: the instruction on the response format to be given to the model. E.g. "The response must be a list of dictionaries, where each dictionary contains the keys 'text' and 'offset'"
            n_shots: the number of examples to provide as few shot prompting   
            offset: whether to require the offset in the response  
            num_proc: the number of processes to use for the parallel processing

        Returns:
            the preprocessed split/layer
        """
        data = data.map(lambda example:  self._apply_to_one_example(example=example, 
                                                                    simplest_prompt=simplest_prompt,
                                                                    instruction_on_response_format = instruction_on_response_format, 
                                                                    offset = offset), 
                        num_proc=num_proc) #batched=True)
        self.offset = offset
        self.instruction_on_response_format = instruction_on_response_format
        self.simplest_prompt = simplest_prompt
        return data

    
    def preprocess_data_one_layer(self, hf_dataset: Dataset, instruction_on_response_format:str='', offset:bool=False, simplest_prompt:bool=False) -> Dataset:
        """
        Preprocess one layer/split of the dataset the trasformations defined in self.apply()

        Args:
            hf_dataset: one layer/split of the dataset to preprocess

        Returns:
            the preprocessed dataset
        """
        if not simplest_prompt and instruction_on_response_format == '':
            raise ValueError("The instruction_on_response_format must be provided when not using the simplest_prompt")
            
        hf_dataset = self.apply(data=hf_dataset, 
                                instruction_on_response_format=instruction_on_response_format, 
                                offset=offset,
                                simplest_prompt=simplest_prompt)
        return hf_dataset
    
    def split_layer_into_train_val_test_(self, dataset: Dataset, split_name: str, test_subset_of_validation: bool=False) -> (Dataset, Dataset):
        """
        Split the layer into train, validation and test sets, according to the split defined at https://github.com/hltfbk/E3C-Corpus/tree/main/documentation

        Args:
            dataset: the dataset to split. Must be a split of the original Hugging Face dataset
            split_name: the name of the layer
            test_subset_of_validation: wether the test set is a subset of the validation set. Set this to True if you want to use the test set as a way of checking on the training throw wandb
                                to mantain the diviosn it train-test of the original repository. Default is False.
        
        Returns:
            the train and test sets
        """
        mapping = {'en.layer1': 'train_labels_en.txt', 
                'es.layer1': 'train_labels_es.txt',
                'eu.layer1': 'train_labels_eu.txt',
                'it.layer1': 'train_labels_it.txt',
                'fr.layer1': 'train_labels_fr.txt',}
        labels_path = mapping[split_name]
        with open(os.path.join('data', labels_path), 'r') as file:
            file_content = file.read()
        labels = file_content.split(", ")
        labels = [label[1:-1] for label in labels]
        idxs_train = [idx for idx, x in enumerate(dataset['original_id']) if x in labels]
        idxs_val = [idx for idx, x in enumerate(dataset['original_id']) if x not in labels]
        random.seed(42)
        idxs_test = random.sample(idxs_val, int(len(idxs_val) * 0.2))
        train_data = dataset.select(idxs_train)
        test_data = dataset.select(idxs_test)
        if test_subset_of_validation:
            val_data = dataset.select(idxs_val)
        else:
            idxs_val = [idx for idx in idxs_val if idx not in idxs_test]
            val_data = dataset.select(idxs_val)

        if self.offset:
            prompt_template = self.prompt_template
        else:
            prompt_template = self.prompt_template_no_offset
        
        def remove_answer_from_prompt(example):
            prompt_no_answ = prompt_template.format(instruction_on_response_format=self.instruction_on_response_format, query=example['sentence'],
                                                    user_start=self.special_tokens_instruction['user_start'],
                                                    user_end=self.special_tokens_instruction['user_end'],
                                                    model_start=self.special_tokens_instruction['model_start'],
                                                    model_end=self.special_tokens_instruction['model_end'])
            example['prompt_with_answer'] = example['prompt']
            example['prompt'] = prompt_no_answ
            return example

        test_data = test_data.map(remove_answer_from_prompt, batched=False)

        return train_data, val_data, test_data
