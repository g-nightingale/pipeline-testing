import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from config import *


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

        X = X.copy()
        X.fillna(self.replacement_value, inplace=True)
        return X

class MeanImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values with the mean.
    """

    def __init__(self):
        self.mean_values = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """
        Find the mean values for each feature in the DataFrame.
        """

        mean_values_dict = {}
        # Only calculate the means for numeric features
        numeric_features = X.select_dtypes(np.number).columns
        for col in numeric_features:
            mean_values_dict[col] = X[col].mean()
        self.mean_values = mean_values_dict

        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replace missing values with the mean of the feature.
        """

        X = X.copy()
        for col in self.mean_values.keys():
            X[col].replace(np.nan, self.mean_values[col], inplace=True)
        return X

class DropFeatures(BaseEstimator, TransformerMixin):
    """
    Drop features specified by the user.
    """

    def __init__(self, drop_features=[]):
        self.drop_features = drop_features

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """
        Don't need to anything.
        """

        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drop the features.
        """

        X = X.copy()
        X.drop(self.drop_features, axis=1, inplace=True)
        return X 

class ConvertCategoricalToOrdinal(BaseEstimator, TransformerMixin):
    """
    Convert categorical variables to ordinal.
    """

    def __init__(self, max_cardinality=100):
        self.mappings = {}
        self.max_cardinality = max_cardinality

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Find categorical variables and create ordinal mappings.
        """
        X = X.copy()
        X['_target'] = y

        mappings = {}
        non_numeric_features = X.select_dtypes(exclude=np.number)
        for feature in non_numeric_features:
                feature_mappings = {}
                if X[feature].value_counts().count() > self.max_cardinality:
                        continue
                else:
                        target_rate_by_feature_value = X.groupby([feature]).agg({'_target':'mean'})
                        target_rate_by_feature_value.sort_values('_target', inplace=True)
                        for i, feature_value in enumerate(target_rate_by_feature_value.index.values):
                                feature_mappings[feature_value] = i
                mappings[feature] = feature_mappings
        self.mappings = mappings

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using the learned mappings.
        """

        X = X.copy()
        for feature in self.mappings.keys():
            X[feature] = X[feature].map(self.mappings[feature])
        return X
