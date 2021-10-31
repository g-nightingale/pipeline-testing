import pytest
import pandas as pd
import os, sys
sys.path.insert(1, os.getcwd())
from src.processing.preprocessing import DropNonNumericFeatures
from src.config import *


def test_drop_non_numeric_features(input_data):
    """Test function drops non-numeric features."""
    
    # Given
    df = input_data

    # When
    dnn = DropNonNumericFeatures()
    dnn.fit(df)
    df = dnn.transform(df)
    
    # Then
    print(dnn.non_numeric_features)
    assert dnn.non_numeric_features == ['col2']
    assert df.columns.to_list() == ['col1', 'col3']



