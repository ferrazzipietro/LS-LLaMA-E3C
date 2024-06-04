from datasets import Dataset, concatenate_datasets
import numpy as np
from tqdm import tqdm
import ml_dtypes


class OutputGenerator():
    def __init__(self, model, tokenizer, label2id, label_list):
        self.model = model
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.label_list = label_list
    
    # def _create_prediction_list(self, model_output):
    #     model_output_logits = model_output.logits.cpu().detach().float().numpy()
    #     preds = np.argmax(model_output_logits, axis=2)
    #     preds_list = []
    #     for pred in preds:
    #         preds_list.append([self.label2id[label] for label in pred])
    #     return preds_list

    def _generate_batch(self, input_sentences):
        encodeds = self.tokenizer(input_sentences, return_tensors="pt", add_special_tokens=False, padding=True)
        #model_inputs = encodeds.to('cuda:0')
        model_inputs = {key: value.to('cuda:0') for key, value in encodeds.items()}
    
        # Print the device of one of the tensors to verify
        print(f'model_inputs is on device: {next(iter(model_inputs.values())).device}')
        generated_ids = self.model(**model_inputs)
        # preds = self._create_prediction_list(generated_ids)
        return generated_ids
    
    def _pad_labels(self, labels, max_length):
        padded_labels = [sublist + [-100] * (max_length - len(sublist)) for sublist in labels]
        return np.array(padded_labels)
            

    def _format_predictions_and_labels(self, generation_output, padded_labels):
        try:
            predictions=generation_output.logits.cpu().detach().numpy()
        except TypeError:
            predictions=generation_output.logits.cpu().detach().float().numpy().astype(ml_dtypes.bfloat16)
        predictions = np.argmax(predictions, axis=2)
        #print('predictions:\n',predictions)
        #print('len(predictions[0]):\n',len(predictions[0]))
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, padded_labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, padded_labels)
        ]
        return {'predictions': true_predictions, 'labels': true_labels}
    
    def generate(self, data, batch_size):
        predictions = []
        labels = []
        total_rows = len(data)
        indexes = [i for i in range(len(data)) if i % batch_size == 0]
        max_index = data.shape[0]

        with tqdm(total=total_rows, desc="generation") as pbar:
            for i, idx in enumerate(indexes[:-1]):
                indici = list(range(idx, indexes[i+1]))
                max_length_sequence =   max(len(seq) for seq in data.select(indici)['labels'])
                padded_labels = self._pad_labels(data.select(indici)['labels'], max_length_sequence)
                generated_output = self._generate_batch(data.select(indici)['sentence'])
                formatted_pred_label = self._format_predictions_and_labels(generated_output, padded_labels)
                predictions.append(formatted_pred_label['predictions'])
                labels.append(formatted_pred_label['labels'])
                pbar.update(batch_size)
            indici = list(range(indexes[len(indexes[:-1])], max_index))
            max_length_sequence =   max(len(seq) for seq in data.select(indici)['labels'])
            padded_labels = self._pad_labels(data.select(indici)['labels'], max_length_sequence)
            generated_output = self._generate_batch(data.select(indici)['sentence'])
            formatted_pred_label = self._format_predictions_and_labels(generated_output, padded_labels)
            predictions.append(formatted_pred_label['predictions'])
            labels.append(formatted_pred_label['labels'])
            pbar.update(batch_size)
        predictions = [item for sublist in predictions for item in sublist]
        labels = [item for sublist in labels for item in sublist]
        predictions = Dataset.from_dict({"predictions": predictions})
        ground_truth_labels = Dataset.from_dict({"ground_truth_labels": labels})
        data = concatenate_datasets([data, predictions, ground_truth_labels], axis=1)
        return data


