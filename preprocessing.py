import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DropNonNumericFeatures(BaseEstimator, TransformerMixin):
    """
    Drop non-numeric features.
    """

    def __init__(self):
        self.non_numeric_features = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """
        Find non-numeric features in a DataFrame.
        """

        self.non_numeric_features = [col for col in X.columns if col not in X.select_dtypes(include=np.number).columns] 
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drop the non-numeric features in a DataFrame.
        """
        
        print(f'features to drop: {self.non_numeric_features}')
        X = X.copy()
        X.drop(self.non_numeric_features, axis=1, inplace=True)
        return X

class ReplaceMissingValues(BaseEstimator, TransformerMixin):
    """
    Replace missing values.
    """

    def __init__(self, replacement_value):
        self.replacement_value = replacement_value

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """
        We don't need the fit function!
        """
 
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replace missing values with the replacement value.
        """

        print(f'missing values will be replaced with: {self.replacement_value}')
        X = X.copy()
        X.fillna(self.replacement_value, inplace=True)
        return X