import pandas as pd
from config import *
from pipeline import loan_pipe
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def create_datasets(train, target):
    """Create datasets."""
    y = train[target]
    X = train.drop(target, axis=1)
    return X, y

def calculate_performance(y_true, y_score):
    """"Calculate performance."""

    return 2 * roc_auc_score(y_true, y_score) - 1

def main():
    """Main function."""

    # Load datasets and split
    train = pd.read_csv(TRAIN_DATA_LOCATION)
    X, y = create_datasets(train, TARGET)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Run pipeline
    loan_pipe.fit(X_train, y_train)

    # Score out train and test sets
    y_train_score = loan_pipe.predict_proba(X_train)[:, 1]
    y_val_score = loan_pipe.predict_proba(X_val)[:, 1]

    # Check performance on train and val
    train_gini = calculate_performance(y_train, y_train_score)
    val_gini = calculate_performance(y_val, y_val_score)

    print(f"train gini: {train_gini}")
    print(f"val gini: {val_gini}")

if __name__ == "__main__":
    main()