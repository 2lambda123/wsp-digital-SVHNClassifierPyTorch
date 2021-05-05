import torch
import torch.utils.data
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torchvision import transforms

from .dataset import Dataset


class Evaluator(object):
    def __init__(self, path_to_lmdb_dir):
        transform = transforms.Compose([
            transforms.CenterCrop([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self._loader = torch.utils.data.DataLoader(Dataset(path_to_lmdb_dir, transform), batch_size=32, shuffle=False)

    def get_model_metrics(self, model):
        y_pred = []
        y_true = []

        with torch.no_grad():
            for batch_idx, (images, length_labels, digits_labels, _) in enumerate(self._loader):
                images, length_labels, digits_labels = images.cuda(), length_labels.cuda(), [digit_labels.cuda() for digit_labels in digits_labels]
                length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits = model.eval()(images)

                # Predictions
                digit1_prediction = digit1_logits.max(1)[1]
                digit2_prediction = digit2_logits.max(1)[1]
                digit3_prediction = digit3_logits.max(1)[1]
                digit4_prediction = digit4_logits.max(1)[1]
                digit5_prediction = digit5_logits.max(1)[1]

                predictions = [
                    digit1_prediction,
                    digit2_prediction,
                    digit3_prediction,
                    digit4_prediction,
                    digit5_prediction,
                ]

                for prediction, label in zip(predictions, digits_labels):
                    for digit_prediction, digit_label in zip(prediction, label):
                        y_pred.append(str(int(digit_prediction)))
                        y_true.append(str(int(digit_label)))

        # Specificity and Sensitivity
        elements = {}
        for true, prediction in zip(y_true, y_pred):
            if true not in elements:
                elements[true] = {"total": 0, "true_positives": 0, "false_positives": 0}
            if prediction not in elements:
                elements[prediction] = {"total": 0, "true_positives": 0, "false_positives": 0}

            elements[true]["total"] += 1
            if prediction == true:
                elements[true]["true_positives"] += 1
            else:
                elements[prediction]["false_positives"] += 1

        for digit in elements:
            elements[digit]["sensitivity"] = elements[digit]["true_positives"] / elements[digit]["total"]

            total_negatives = len(y_true) - elements[digit]["total"]
            elements[digit]["specificity"] = (total_negatives - elements[digit]["false_positives"]) / total_negatives

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")

        matrix = confusion_matrix(y_true, y_pred,
                                  labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])

        report = classification_report(y_true, y_pred, output_dict=True, digits=2)
        # Round report to 2 decimal places
        for label, metrics in report.items():
            if label in elements:
                report[label]["specificity"] = elements[label]["specificity"]
                report[label]["sensitivity"] = elements[label]["sensitivity"]

            if label != "accuracy":
                for metric, value in metrics.items():
                    metrics[metric] = round(value, 2)

        # Add row/column totals to confusion matrix
        matrix = matrix.tolist()
        col_sums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for row in matrix:
            row.append(sum(row))
            for idx, col in enumerate(row):
                col_sums[idx] += col
        matrix.append(col_sums)

        model_info = {
            "accuracy": round(accuracy, 2),
            "confusion_matrix": matrix,
            "classification_report": report,
            "precision": round(precision, 2),
            "f1_score": round(f1, 2),
            "recall": round(recall, 2),
        }
        return model_info

    def evaluate(self, model):
        num_correct = 0
        needs_include_length = False

        with torch.no_grad():
            for batch_idx, (images, length_labels, digits_labels, _) in enumerate(self._loader):
                images, length_labels, digits_labels = images.cuda(), length_labels.cuda(), [digit_labels.cuda() for digit_labels in digits_labels]
                length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits = model.eval()(images)

                if batch_idx == 0:
                    details = [length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits, length_labels, digits_labels]

                length_prediction = length_logits.max(1)[1]
                digit1_prediction = digit1_logits.max(1)[1]
                digit2_prediction = digit2_logits.max(1)[1]
                digit3_prediction = digit3_logits.max(1)[1]
                digit4_prediction = digit4_logits.max(1)[1]
                digit5_prediction = digit5_logits.max(1)[1]

                if needs_include_length:
                    num_correct += (length_prediction.eq(length_labels) &
                                    digit1_prediction.eq(digits_labels[0]) &
                                    digit2_prediction.eq(digits_labels[1]) &
                                    digit3_prediction.eq(digits_labels[2]) &
                                    digit4_prediction.eq(digits_labels[3]) &
                                    digit5_prediction.eq(digits_labels[4])).cuda().sum()
                else:
                    num_correct += (digit1_prediction.eq(digits_labels[0]) &
                                    digit2_prediction.eq(digits_labels[1]) &
                                    digit3_prediction.eq(digits_labels[2]) &
                                    digit4_prediction.eq(digits_labels[3]) &
                                    digit5_prediction.eq(digits_labels[4])).cuda().sum()

        accuracy = num_correct.item() / len(self._loader.dataset)
        return accuracy, details
