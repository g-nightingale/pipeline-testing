import pandas as pd

def main():
    """Main function."""

    train = pd.read_csv("data/train.csv")

    print(train.head())

if __name__ == "__main__":
    main()