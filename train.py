from pathlib import Path

import numpy as np

from data_utils import load_dataset
from metrics import format_results_table, print_metrics, summarize_metrics
from models import get_model


RANDOM_SEED = 42
RESULTS_DIR = Path("results")


def fit_and_evaluate(model_name, data_bundle):
    model = get_model(model_name, random_state=RANDOM_SEED)
    model.fit(data_bundle.x_train, data_bundle.y_train)

    val_pred = model.predict(data_bundle.x_val)
    test_pred = model.predict(data_bundle.x_test)

    val_metrics = summarize_metrics(data_bundle.y_val, val_pred)
    test_metrics = summarize_metrics(data_bundle.y_test, test_pred)

    return {
        "model": model,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def save_results(results):
    RESULTS_DIR.mkdir(exist_ok=True)

    output_path = RESULTS_DIR / "metrics_summary.txt"
    output_path.write_text(format_results_table(results), encoding="utf-8")
    print(f"\nSaved metrics summary to: {output_path}")


def train_decision_tree(data_bundle):
    """
    Train the baseline model that keeps the project runnable before teammates
    complete the boosting methods.
    """
    return fit_and_evaluate("decision_tree", data_bundle)

def train_adaboost(data_bundle):
    """
    Train AdaBoost and return val/test metrics.
    """
    return fit_and_evaluate("adaboost", data_bundle) 

def train_gradient_boosting(data_bundle):
    """
    Train and evaluate the Gradient Boosting model.

    """
    return fit_and_evaluate("gradient_boosting", data_bundle)


def main():
    np.random.seed(RANDOM_SEED)
    data_bundle = load_dataset(random_state=RANDOM_SEED)

    print("Boosting final project scaffold")
    print(f"Dataset: {data_bundle.dataset_name}")
    print(f"x_train: {data_bundle.x_train.shape}")
    print(f"x_val:   {data_bundle.x_val.shape}")
    print(f"x_test:  {data_bundle.x_test.shape}")
    print()

    results = {}

    baseline_output = train_decision_tree(data_bundle)
    print_metrics("decision_tree validation", baseline_output["val_metrics"])
    print_metrics("decision_tree test", baseline_output["test_metrics"])
    print()
    results["decision_tree"] = baseline_output["test_metrics"]

    adaboost_output = train_adaboost(data_bundle)
    print_metrics("adaboost validation", adaboost_output["val_metrics"])
    print_metrics("adaboost test", adaboost_output["test_metrics"])
    print()
    results["adaboost"] = adaboost_output["test_metrics"]

    gradient_output = train_gradient_boosting(data_bundle)
    print_metrics("gradient_boosting validation", gradient_output["val_metrics"])
    print_metrics("gradient_boosting test", gradient_output["test_metrics"])
    print()
    results["gradient_boosting"] = gradient_output["test_metrics"]

    save_results(results)


if __name__ == "__main__":
    main()
