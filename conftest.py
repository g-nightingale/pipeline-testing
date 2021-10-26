import pytest
import pandas

@pytest.fixture
def input_data():
    df = pd.DataFrame({
                        'col1': [1, 0],
                        'col2': ['a', 'b'],
                        'col3': [0, 1]
                    })
    return df