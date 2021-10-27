import pytest
import pandas as pd
from preprocessing import DropFeatures
from config import *

def test_replace_missing_values(input_data):
    """Test function replaces missing values."""
    
    # Given
    df = input_data

    # When
    dp = DropFeatures(drop_features='col2')
    df = dp.transform(df)
    
    # Then
    assert df.columns.to_list() == ['col1', 'col3']
