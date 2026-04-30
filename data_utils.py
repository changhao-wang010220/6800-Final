from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42
DATA_DIR = Path("data")
DATASET_PATH = DATA_DIR / "breast_cancer.csv"


@dataclass
class DatasetBundle:
    x_train: object
    x_val: object
    x_test: object
    y_train: object
    y_val: object
    y_test: object
    feature_names: list[str]
    target_names: list[str]
    dataset_name: str


def load_dataset(test_size=0.2, val_size=0.2, random_state=RANDOM_SEED):
    """
    Load the local CSV dataset used for the boosting study.

    The CSV is stored in data/breast_cancer.csv so the project keeps a
    concrete dataset file in the repo for teammates to inspect directly.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Expected local dataset at {DATASET_PATH}, but it was not found."
        )

    df = pd.read_csv(DATASET_PATH)
    feature_names = [column for column in df.columns if column != "label"]
    x = df[feature_names].to_numpy()
    y = df["label"].to_numpy()

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_full,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    return DatasetBundle(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=feature_names,
        target_names=["malignant", "benign"],
        dataset_name="breast_cancer",
    )
