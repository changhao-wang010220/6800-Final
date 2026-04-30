import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def accuracy(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))


def balanced_accuracy(y_true, y_pred):
    return float(balanced_accuracy_score(y_true, y_pred))


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro"))


def summarize_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
    }


def print_metrics(model_name, metric_dict):
    print(f"{model_name}")
    print(f"  Accuracy: {metric_dict['accuracy']:.4f}")
    print(f"  Balanced accuracy: {metric_dict['balanced_accuracy']:.4f}")
    print(f"  Macro F1-score: {metric_dict['macro_f1']:.4f}")


def format_results_table(results):
    lines = []
    header = (
        f"{'Model':<20}"
        f"{'Accuracy':>12}"
        f"{'Balanced Acc':>16}"
        f"{'Macro F1':>12}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for model_name, metric_dict in results.items():
        lines.append(
            f"{model_name:<20}"
            f"{metric_dict['accuracy']:>12.4f}"
            f"{metric_dict['balanced_accuracy']:>16.4f}"
            f"{metric_dict['macro_f1']:>12.4f}"
        )

    return "\n".join(lines)
