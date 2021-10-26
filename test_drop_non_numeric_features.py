import pytest
import pandas as pd
from preprocessing import DropNonNumericFeatures

@pytest.fixture
def input_data():
    df = pd.DataFrame({
                        'col1': [1, 0],
                        'col2': ['a', 'b'],
                        'col3': [0, 1]
                    })
    return df

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



