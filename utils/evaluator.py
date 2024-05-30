import pandas as pd
from datasets import Dataset

class Evaluator():
    def __init__(self, data, tokenizer):
        self.data = data
        # self.evaluation_table = pd.DataFrame(columns=['TP', 'FP', 'FN'])
        self.evaluation_table = {}
        self.tokenizer = tokenizer
        
     
    def _compare_prediction_label_one_example_token_by_token(self, example) -> (int, int, int):
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
        # labels = ['O'] + labels[:-1] 
        for pred, lab in zip(predictions, labels):
            TP = TP + (1 if pred != 'O' and lab != 'O' else 0)
            FP = FP + (1 if pred != lab and lab =='O' else 0)
            FN = FN + (1 if pred != lab and pred =='O' else 0)
        TN = len(predictions) - (TP + FP + FN) 
        try:
            precision = TP / (TP + FP)
        except:
            precision = 0
        try:
            recall = TP / (TP + FN)
        except:
            recall = 0
        try:
            f1 = 2 * (precision * recall) / (precision + recall)
        except:
            f1 = 0
        
        example['TP'] = TP
        example['FP'] = FP
        example['FN'] = FN
        example['TN'] = TN
        example['precision'] = precision
        example['recall'] = recall
        example['f1'] = f1
        return example
    
    def extract_FP_FN_TP_TN_token_by_token(self) -> (int, int, int):
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
        self.data = self.data.map(self._compare_prediction_label_one_example_token_by_token, batched=False)


    def create_evaluation_table(self, input_data:Dataset = None):
        """
        Create the evaluation table with the number of False Positives, False Negatives and True Positives.

        Args:
        input_data (Dataset): the dataset to evaluate. If None, the data used to initialize the class is used.

        Returns:
        dict: the evaluation table with the number of False Positives, False Negatives and True Positives.
        """
        if input_data is not None:
            tmp_data = input_data
        else:
            tmp_data = pd.DataFrame(self.data)
        TP = tmp_data['TP'].sum()
        FP = tmp_data['FP'].sum()
        FN = tmp_data['FN'].sum()
        # TN = tmp_data['TN'].sum()
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        self.evaluation_table = {'TP': TP, 'FP': FP, 'FN': FN, #'TN': TN,
                                  'precision':precision, 'recall':recall, 'f1':f1}
        
        return self.evaluation_table
    
