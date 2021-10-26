import pandas as pd
from config import *
from pipeline import loan_pipe
from sklearn.model_selection import train_test_split

def create_datasets(train, target):
    """Create datasets."""
    y = train[target]
    X = train.drop(target, axis=1)
    return X, y

def main():
    """Main function."""

    # Load datasets and split
    train = pd.read_csv(TRAIN_DATA_LOCATION)
    X, y = create_datasets(train, TARGET)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Run pipeline
    loan_pipe.fit(X_train, y_train)

if __name__ == "__main__":
    main()