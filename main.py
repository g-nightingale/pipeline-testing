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

def calculate_performance(y_train_true, y_train_score, y_test_true, y_test_score):
    """"Calculate performance."""
    def gini(y_true, y_score):
        return 2 * roc_auc_score(y_true, y_score) - 1

    train_gini = gini(y_train_true, y_train_score)
    test_gini = gini(y_test_true, y_test_score)

    print(f"train gini: {train_gini}")
    print(f"test gini: {test_gini}")

    return train_gini, test_gini

def main():
    """Main function."""

    # Load datasets and split
    train = pd.read_csv(TRAIN_DATA_LOCATION)
    X, y = create_datasets(train, TARGET)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Run pipeline
    loan_pipe.fit(X_train, y_train)

    # Score out train and test sets
    y_train_score = loan_pipe.predict_proba(X_train)[:, 1]
    y_test_score = loan_pipe.predict_proba(X_test)[:, 1]

    # Check performance on train and test
    train_gini, test_gini = calculate_performance(y_train, y_train_score, y_test, y_test_score)

if __name__ == "__main__":
    main()