import pytest
import pandas as pd
import os, sys
sys.path.insert(1, os.getcwd())
from src.processing.preprocessing import ConvertCategoricalToOrdinal
from src.config import *


def test_convert_categorical_to_ordinal(input_data_categorical_with_target):
    """Test function replaces missing values."""
    
    # Given
    df = input_data_categorical_with_target
    X = df[['col1', 'col2']]
    y = df['col3']

    # When
    cc = ConvertCategoricalToOrdinal()
    X = cc.fit_transform(X, y)
    
    # Then
    assert list(X['col1']) == [1, 0, 2]
    assert len(X.columns) == 2
    



