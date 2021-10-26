import pytest
import pandas as pd
from preprocessing import DropNonNumericFeatures, ReplaceMissingValues
from config import *

def test_replace_missing_values(input_data_with_missing_values):
    """Test function replaces missing values."""
    
    # Given
    df = input_data_with_missing_values

    # When
    rmv = ReplaceMissingValues(REPLACE_MISSING_VALUE)
    df = rmv.transform(df)
    
    # Then
    assert df.iloc[0, 0] == REPLACE_MISSING_VALUE
    assert df.iloc[0, 0] == REPLACE_MISSING_VALUE
    



