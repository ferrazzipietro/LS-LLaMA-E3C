from utils.data_preprocessor import DataPreprocessor
from datasets import Dataset
from tqdm import tqdm

class TestDataProcessor():
    def __init__(self, test_data: Dataset, preprocessor:DataPreprocessor, n_shots_inference:int, language:str, tokenizer) -> None:
        """
        Initialize the TestDataProcessor class.
        pass to this the same DataPreprocessor used for the training data. This will ensure that the inference prompt is formatted in the same way as the training prompt.
        """
        self.test_data = test_data
        self.preprocessor = preprocessor
        self.language = language
        self.tokenizer = tokenizer
        self.model_type = preprocessor.model_type
        self.few_shots_dict = {'en':{'questions':['We present a case of a 32-year-old woman with a history of gradual enlargement of the anterior neck.',
                                                   'Patient information: a 9-month-old boy presented to the emergency room with a 3-day history of refusal to bear weight on the right lower extremity and febrile peaks of up to 38.5°C for 24 hours.',
                                                   'There was no evidence of lung lesions.',
                                                   'Locally diminished actin coloration indicated atrophy of smooth muscle fibers.'],
                                        'responses':['[{"entity": "present"}, {"entity": "history"}, {"entity": "enlargement"}]',
                                                     '[{"entity": "presented"}, {"entity": "refusal"}, {"entity": "bear"}, {"entity": "peaks"}]',
                                                      '[{"entity": "evidence"}, {"entity": "lung lesions"]',
                                                      '[{"entity": "coloration"}, {"entity": "indicated"}, {"entity": "atrophy"}, {"entity": "atrophy of smooth muscle fibers"}, {"entity": "smooth muscle fibers"'],
                                        'responses_offset': ['[{"entity": "present", "offset": [3, 10]}, {"entity": "history", "offset": [48, 55]}, {"entity": "enlargement", "offset": [67, 78]}]',
                                                             '[{"entity": "presented", "offset": [39, 48]}, {"entity": "refusal", "offset": [95, 102]}, {"entity": "bear", "offset": [106, 110]}, {"entity": "peaks", "offset": [159, 164]}]',
                                                             '[{ "entity": "evidence", "offsets": [13, 21]}, {"entity": "lung lesions", "offsets": [25, 37]} ]',
                                                             '[{"entity": "coloration", "offsets": [25, 35]}, {"entity": "indicated", "offsets": [36, 45]}, {"entity": "atrophy","offsets": [46, 53]}, {"entity": "atrophy of smooth muscle fibers", "offsets": [46, 77]}, {"entity": "smooth muscle fibers", "offsets": [57, 77]} ]'],
                                    },
                                'it':{'questions':['In considerazione dell’inefficacia della terapia somministrata, in assenza di ulteriori opzioni terapeutiche standard potenzialmente efficaci e dopo colloquio con i genitori si decide di avviare la paziente a trapianto aploidentico, possibilmente NK allo reattivo, da genitore.',
                                                    'L’esame istologico dimostrava mucosa gastrica atrofica con flogosi cronica, marcato edema ed incremento del connettivo del corion, focale metaplasia intestinale, il tutto sovrastante un tessuto fibromuscolare.',
                                                    'Giunge nel nostro reparto per stranguria in assenza di altri sintomi.',
                                                    'All’età di 16 mesi, nuovo ricovero per febbre (39°C) e stato di abbattimento.'],
                                       'responses':['[{"entity": "inefficacia"}, {"entity": "opzioni"}, {"entity": "colloquio"}, {"entity": "avviare"}, {"entity": "trapianto"}, {"entity": "genitori"}, {"entity": "paziente"}, {"entity": "genitore"}]',
                                                    '[{"entity": "mucosa gastrica atrofica"}, {"entity": "flogosi\r\cronica"}]',
                                                    '[{"entity": "Giunge"}, {"entity": "stranguria"}, {"entity": "sintomi"}, {"entity": "stranguria"}]',
                                                    '[{"entity": "ricovero"}, {"entity": "febbre"}, {"entity": "stato"}, {"entity": "febbre"}, {"entity": "39°C"} ]'],
                                       'responses_offset':['[{"entity": "inefficacia", "offset": [23, 34]}, {"entity": "opzioni", "offset": [88,95]}, {"entity": "colloquio", "offset": [149,158]}, {"entity": "avviare", "offset": [187,194]}, {"entity": "trapianto", "offset": [209,218]}, {"entity": "genitori", "offset": [163,173]}, {"entity": "paziente", "offset": [195,106]}, {"entity": "genitore", "offset": [268,276]}]',
                                                           '[{"entity": "mucosa gastrica atrofica", "offset": [30,54]}, {"entity": "flogosi\r\cronica", "offset": [59,75]}]',
                                                           '[{"entity": "Giunge", "offset": [0,6]}, {"entity": "stranguria", "offset": [30,40]}, { "entity": "sintomi", "offset": [61,68]}, {"entity": "stranguria", "offset": [ 30, 40 ]} ]',
                                                           '[{"entity": "ricovero", "offset": [26,34]}, {"entity": "febbre", "offset": [ 39, 45 ]}, {"entity": "stato", "offset": [ 55, 60 ]}, {"entity": "febbre", "offset": [ 39, 45 ]}, {"entity": "39°C", "offset": [47,51]} ]']}
                                }
        if len(self.few_shots_dict[self.language]['questions']) < n_shots_inference:
            raise ValueError(f'The number of shots for the inference prompt is greater than the number of examples available.')
        if len(self.few_shots_dict[self.language]['responses']) < n_shots_inference:
            raise ValueError(f'The number of shots for the inference prompt is greater than the number of responses available.')
        self.n_shots_inference = n_shots_inference
    
    def _extract_ground_truth(self, prompt:str) -> str:
        # print('PROMPT: ', prompt)
        end_of_prompt_string = self.preprocessor.special_tokens_instruction['user_end'] + self.preprocessor.special_tokens_instruction['model_start']
        # print('end_of_prompt_string: ', end_of_prompt_string)
        out = prompt.split(end_of_prompt_string, 1)
        out = out[1].strip().replace(self.preprocessor.special_tokens_instruction['model_start'], '').replace(self.preprocessor.special_tokens_instruction['model_end'], '')
        # print('OUT: ', out)
        return {'ground_truth': out}
    
    def _format_prompt_inference(self, input: str, instruction_on_response_format:str, n_shots:int, offset: bool, simplest_prompt:bool, output:str='', list_of_examples: [str]=[], list_of_responses:[str]=[]) -> str:
        """
        Format the input and output into a prompt for the finetuning

        Args:
            task: the task for which the prompt is generated, either 'finetuning' or 'inference'
            input: the input text
            instruction_on_response_format: the instruction on the response format. E.g. "The response must be a list of dictionaries, where each dictionary contains the keys 'text' and 'offset'"
            n_shots: the number of examples to provide as few shot prompting
            offset: whether to require the offset in the response
            tokenizer: the tokenizer to use
            output: the output text
            list_of_examples: the list of examples to provide as few shot prompting
            list_of_responses: the list of responses to provide as few shot prompting

        Returns:
            the formatted prompt
        """
        if output != '':
            raise ValueError("The output must be an empty string when generating prompts for the inference")

        if len(list_of_examples) != len(list_of_responses):
            raise ValueError("The number of examples and responses must be the same")
        if n_shots != len(list_of_examples):
            raise ValueError("The number of examples and shots must be the same")
        if n_shots != len(list_of_responses):
            raise ValueError("The number of responses and shots must be the same")
        
        if simplest_prompt:
            base_prompt = self.preprocessor._simplest_base_prompt_input(input)
        elif not simplest_prompt:
            base_prompt = self.preprocessor._base_prompt_input(input, instruction_on_response_format)

        one_shot_example = self.preprocessor.one_shot_example_no_offset if not offset else self.preprocessor.one_shot_example
            
        prompt = ''
        for shot_example in range(n_shots):
            prompt += one_shot_example.format(
                instruction_on_response_format=instruction_on_response_format, 
                example_query=list_of_examples[shot_example], 
                example_response=list_of_responses[shot_example],
                user_start=self.preprocessor.special_tokens_instruction['user_start'],
                user_end=self.preprocessor.special_tokens_instruction['user_end'],
                model_start=self.preprocessor.special_tokens_instruction['model_start'],
                model_end=self.preprocessor.special_tokens_instruction['model_end'])
        
        bos_token = self.preprocessor.tokenizer.bos_token
        if self.model_type == 'qwen':
            bos_token = ''
        prompt = bos_token + prompt + base_prompt + output 
                            
        return prompt
    
    def _extract_inference_prompt(self, sentence:str, simplest_prompt:bool) -> str:
        if self.preprocessor.offset:
            few_shots_responses = self.few_shots_dict[self.language]['responses_offset']
        else:
            few_shots_responses = self.few_shots_dict[self.language]['responses']
        if self.n_shots_inference == 0:
            list_of_examples = []
            list_of_responses = []
        else:
            list_of_examples = self.few_shots_dict[self.language]['questions'][0:self.n_shots_inference]
            list_of_responses = few_shots_responses[0:self.n_shots_inference]
        inference_prompt = self._format_prompt_inference(input=sentence, 
                                                        instruction_on_response_format=self.preprocessor.instruction_on_response_format,
                                                        offset=self.preprocessor.offset,
                                                        output='',
                                                        n_shots=self.n_shots_inference,
                                                        simplest_prompt=simplest_prompt,
                                                        list_of_examples=list_of_examples,
                                                        list_of_responses=list_of_responses)
        return {'inference_prompt': inference_prompt}
    
    def add_inference_prompt_column(self, simplest_prompt:bool) -> None:
        """
        Add the inferencePrompt and groundTruth columns to the test_data dataframe.
        """
        self.test_data = self.test_data.map(lambda x: self._extract_inference_prompt(x['sentence'], simplest_prompt=simplest_prompt))
    
    def add_ground_truth_column(self) -> None:
        """
        Add the groundTruth column to the test_data dataframe.
        """
        self.test_data = self.test_data.map(lambda x: self._extract_ground_truth(x['prompt']))

    def _generate_model_response(self, examples, model, tokenizer, max_new_tokens_factor:float) -> str:
        device = "cuda"
        tokenizer.padding_side = "left"
        # if self.model_type == 'qwen':
        #     tokenizer.pad_token = '<unk>' # tokenizer.special_tokens['<extra_0>']
        input_sentences = examples['sentence']
        prompts = examples['inference_prompt']
        input_sentences_tokenized = tokenizer(input_sentences, return_tensors="pt", padding=True)
        max_new_tokens = int(len(max(input_sentences_tokenized, key=len)) * max_new_tokens_factor)
        # if self.preprocessor.model_type == 'gemma':
        #     add_special_tokens = True
        encodeds = tokenizer(prompts, return_tensors="pt", add_special_tokens=False, padding=True)
        model_inputs = encodeds.to(device)
        generated_ids = model.generate(**model_inputs, do_sample=True, max_new_tokens=max_new_tokens,  
                                       pad_token_id=tokenizer.pad_token_id,
                                       temperature = 1.0) # max_new_tokens=max_new_tokens,
        generated_ids = generated_ids[:, encodeds.input_ids.shape[1]:]
        decoded = tokenizer.batch_decode(generated_ids)
        # decoded = [self._postprocess_model_output(i) for i in decoded]
        return (decoded)
                
    def add_responses_column(self, model, tokenizer, batch_size:int, max_new_tokens_factor:float) -> None:
        """
        Adds a column with the response of the model to the actual query.
        
        params:
        model: the model to use to generate the response
        tokenizer: the tokenizer to use to generate the response
        batch_size: the batch size to use to process the examples. Increasing this makes it faster but requires more GPU. Default is 8.
        max_new_tokens_factor: the factor conotrolling the number of new tokens to generate. This is a factor of the length of the input sentence.
        """
        responses_col = []
        total_rows = len(self.test_data)
        indexes = [i for i in range(len(self.test_data)) if i % batch_size == 0]
        max_index = self.test_data.shape[0]


        with tqdm(total=total_rows, desc="generating responses") as pbar:
            for i, idx in enumerate(indexes[:-1]):
                indici = list(range(idx, indexes[i+1]))
                tmp = self._generate_model_response(self.test_data.select(indici), model, tokenizer, max_new_tokens_factor)
                responses_col.extend(tmp)
                pbar.update(batch_size)
            indici = list(range(indexes[len(indexes[:-1])], max_index))
            tmp = self._generate_model_response(self.test_data.select(indici), model, tokenizer, max_new_tokens_factor)
            responses_col.extend(tmp)
            pbar.update(batch_size)

        self.test_data = self.test_data.add_column('model_responses', responses_col)
    
    def _postprocess_model_output_deprecated(self, model_output: str) -> str:
        """
        Postprocess the model output to remove the instruction and return the model response.

        Args:
        model_output (str): the model output as it is returned by the model. The processing of the output is done in the function

        return:
        str: the model response, i.e. the model output without the instruction

        """
        end_of_prompt_string = self.preprocessor.special_tokens_instruction['user_end'] + self.preprocessor.special_tokens_instruction['model_start']
        return model_output.split(end_of_prompt_string, 1)[-1].strip()