from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from config import REPLACE_MISSING_VALUE
import preprocessing as pp

loan_pipe = Pipeline(
    [
        (
            "drop_non_numeric_features",
            pp.DropNonNumericFeatures(),
        ),
        (
            "replace_missing_values",
            pp.ReplaceMissingValues(replacement_value=REPLACE_MISSING_VALUE),
        ),
        (
            "gb_model",
            LogisticRegression(),
        ),
    ]
)
