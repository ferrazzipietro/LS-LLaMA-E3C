import string


class DatasetFormatConverter():
    """
    """
    def __init__(self, dataset, clent:bool=False):
        self.dataset = dataset
        self.label2id = { "O": 0, "B": 1, "I": 2}
        self.clent = clent

    def get_id2label(self):
        id2label = {v: k for k, v in self.label2id.items()}
        return id2label
    
    def get_label2id(self):
        return self.label2id
    
    def get_label_list(self):
        return list(self.label2id.keys())
    
    def _reformat_entities_dict(self, enitities_dicts_list):
        return [{item.get('text') : item.get('offsets')} for item in enitities_dicts_list]
    
    def _generate_char_based_labels_list(self, example):
        labels = ["O"] * len(example["sentence"])
        for entity in example['entities']:
            # print('entity: ', entity)
            start = entity["offsets"][0]
            end = entity["offsets"][1]
            type = entity["type"]
            labels[start] = f"B-{type}"
            for i in range(start+1, end):
                # print('char: ', example["sentence"][i])
                labels[i] = f"I-{type}"
        return labels
    
    def _contains_punctuation(self, word):
        return any(char in string.punctuation for char in word)

    def _is_only_punctuation(self, word):
        return all(char in string.punctuation for char in word)
    
    def _remove_punctuation_and_count(self, text, punctuation_to_remove = '!"#&\'(),-./:;<=>?@[\\]^_`|'):
        """
        Remove punctuation from the beginning and end of the text and count how many characters were removed.
        """
        count_beginning = len(text) - len(text.lstrip(punctuation_to_remove))
        count_end = len(text) - len(text.rstrip(punctuation_to_remove))
        word_no_punct = text.strip(punctuation_to_remove)
        return word_no_punct, count_beginning, count_end
    
    def _only_clent_from_enities(self, example):
        example['entities'] = [entity for entity in example['entities'] if entity['type'] == 'CLINENTITY']
        return example
    
    def apply_only_clinentities(self):
        self.dataset = self.dataset.map(self._only_clent_from_enities, num_proc=1)

    def _entities_from_dict_to_labels_list(self, example, word_level=True, token_level=False, tokenizer=None):
        if word_level and token_level:
            raise ValueError("Only one of word_level and token_level can be True")
        if not word_level and not token_level:
            raise ValueError("One of word_level and token_level must be True")
        if token_level and tokenizer is None:
            raise ValueError("tokenizer must be provided if token_level is True")
        if word_level:
            words = example["sentence"].split()
        elif token_level:
            raise NotImplementedError
        labels = [0] * len(words)
        # print(example["entities"])
        chars_based_labels = self._generate_char_based_labels_list(example)
        word_starting_position = 0
        for i, word in enumerate(words):
            # print(f'processing word: {word}\n starting position: {word_starting_position}\n encompassing labels {chars_based_labels[word_starting_position:word_starting_position+len(word)]}')
            if self._is_only_punctuation(word):
                word_starting_position = word_starting_position + len(word) + 1
                continue
            if self._contains_punctuation(word):
                _, count_beginning, count_end = self._remove_punctuation_and_count(word)
                # print(f'remove punctuation from word: {word}\n count beginning: {count_beginning}\n count end: {count_end}')
            else:
                count_beginning, count_end = 0, 0
            word_length = len(word)
            start_word = word_starting_position + count_beginning
            end_word = word_starting_position + word_length - count_end
            chars_labels_of_this_word = chars_based_labels[start_word : end_word]
            if (chars_labels_of_this_word[0].startswith("B-") or chars_labels_of_this_word[0].startswith("I-")) \
                and all([label.startswith("I-") for label in chars_labels_of_this_word[1:]]):
                labels[i] = self.label2id.get(chars_labels_of_this_word[0][0], -1)
            word_starting_position = word_starting_position + word_length + 1
        # print(labels)
        example['words'] = words
        example['word_level_labels'] = labels
        return example

    def apply(self):
        if self.clent:
            self.apply_only_clinentities()
        self.dataset = self.dataset.map(self._entities_from_dict_to_labels_list)
        self.dataset = self.dataset.rename_column("word_level_labels", "ner_tags")
        self.dataset = self.dataset.rename_column("words", "tokens")

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def set_max_seq_length(self, max_seq_length):
        self.max_seq_length = max_seq_length

    # def tokenize_and_align_labels(self, examples): COPIED FROM HF, WRONG
    #     """
    #     """
    #     tokenized_inputs = self.tokenizer(examples["tokens"], is_split_into_words=True, padding='longest', max_length=self.max_seq_length, truncation=True)

    #     labels = []
    #     for i, label in enumerate(examples[f"ner_tags"]):
    #         word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
    #         previous_word_idx = None
    #         label_ids = []
    #         for word_idx in word_ids:  # Set the special tokens to -100.
    #             if word_idx is None:
    #                 label_ids.append(-100)
    #             elif word_idx != previous_word_idx:  # Only label the first token of a given word.
    #                 label_ids.append(label[word_idx])
    #             else:
    #                 label_ids.append(-100)
    #             previous_word_idx = word_idx
    #         labels.append(label_ids)
    #     tokenized_inputs["labels"] = labels
    #     return tokenized_inputs

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, add_special_tokens=False)
        labels = []
        for i, words_label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            label_ids = []
            for k, word_idx in enumerate(word_ids): 
                same_word_as_previous  = False if (word_idx != word_ids[k-1] or k==0) else True
                if word_idx is None:
                    token_label = -100
                elif words_label[word_idx] == self.label2id['O']:
                    token_label = self.label2id['O']
                elif same_word_as_previous:
                    token_label = self.label2id['I']
                elif not same_word_as_previous:
                    token_label = words_label[word_idx]
                label_ids.append(token_label)
                # if word_idx is not None:#  and k>12:
                #     print("word_label: ", words_label[word_idx])
                # print(tokenizer.decode(tokenized_inputs[i].ids[k]), ": ",word_idx,  "\nassigned_token_label:",  label_ids[k], '\n')
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
        