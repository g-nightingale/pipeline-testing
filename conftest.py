import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def input_data():
    df = pd.DataFrame({
                        'col1': [1, 0],
                        'col2': ['a', 'b'],
                        'col3': [0, 1]
                    })
    return df

@pytest.fixture
def input_data_with_missing_values():
    df = pd.DataFrame({
                        'col1': [np.nan, 0, 2],
                        'col2': ['a', 'b', 'c'],
                        'col3': [0, np.nan, 4]
                    })
    return df