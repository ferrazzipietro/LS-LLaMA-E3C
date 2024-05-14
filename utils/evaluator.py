import pandas as pd

class Evaluator():
    def __init__(self, data, tokenizer):
        self.data = data
        # self.evaluation_table = pd.DataFrame(columns=['TP', 'FP', 'FN'])
        self.evaluation_table = {}
        self.tokenizer = tokenizer
        
     
    def _compare_prediction_label_one_example(self, example) -> (int, int, int):
        """
        Compare the prediction with the label of one sentence.
        Args:
        predictions (list[str]): the list of the predicted labels
        labels (list[str]): the list of the true labels
        return:
        int: the number of false positives
        int: the number of false negatives
        int: the number of true positives
        """
        predictions = example['predictions']
        labels = example['ground_truth_labels']
        TP, FP, FN = 0, 0, 0
        for pred, lab in zip(predictions, labels):
            TP = TP + (1 if pred == lab and lab!='O' else 0)
            FP = FP + (1 if pred != lab and lab =='O' else 0)
            FN = FN + (1 if pred != lab and pred =='O' else 0)
        example['TP'] = TP
        example['FP'] = FP
        example['FN'] = FN

        return example
    
    def extract_FP_FN_TP(self) -> (int, int, int):
        """
        Extract the number of False Positives, False Negatives and True Positives from the model output and the ground truth.
        Args:
        predictions (list[str]): the list of the predicted labels
        labels (list[str]): the list of the true labels
        return:
        int: the number of false positives
        int: the number of false negatives
        int: the number of true positives
        """
        self.data = self.data.map(self._compare_prediction_label_one_example, batched=False)

    def create_evaluation_table(self):
        tmp_data = pd.DataFrame(self.data)
        TP = tmp_data['TP'].sum()
        FP = tmp_data['FP'].sum()
        FN = tmp_data['FN'].sum()
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        self.evaluation_table = {'TP': TP, 'FP': FP, 'FN': FN,
                                  'precision':precision, 'recall':recall, 'f1':f1}
        
        return self.evaluation_table
    
    def print_disallined_Is(self):
        """
        """
        counter = 0
        tot_tokens = 0
        for example in self.data:
            sentence_pred = example['predictions']
            sentence = example['sentence']
            previous = '' 
            append = False
            for token in sentence_pred:
                if token=='I' and previous=='O':
                    append = True
                    counter+=1
                    # print('token:', token, 'previous:', previous, 'position:', i, 'sentence:', sentence_pred)
                previous = token
                tot_tokens += 1
            if append:
                tokenized_input = self.tokenizer(sentence)
                tokens = self.tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
                print([(t,p, ground_truth_label) for t, p, ground_truth_label in zip(tokens, sentence_pred, example['ground_truth_labels'])])