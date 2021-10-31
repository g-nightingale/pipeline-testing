import pytest
import pandas as pd
import os, sys
sys.path.insert(1, os.getcwd())
from src.processing.preprocessing import ReplaceMissingValues
from src.config import *


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
