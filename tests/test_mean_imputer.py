import pytest
import pandas as pd
import os, sys
sys.path.insert(1, os.getcwd())
from src.processing.preprocessing import MeanImputer
from src.config import *


def test_replace_missing_values(input_data_with_missing_values):
    """Test function replaces missing values."""
    
    # Given
    df = input_data_with_missing_values

    # When
    mi = MeanImputer()
    mi.fit(df)
    df = mi.transform(df)
    
    # Then
    assert mi.mean_values['col1'] == 1
    assert df.iloc[0, 0] == 1
    assert mi.mean_values['col3'] == 2
    assert df.iloc[1, 2] == 2

    



