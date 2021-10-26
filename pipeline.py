from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import preprocessing as pp

loan_pipe = Pipeline(
    [
        (
            "drop_non_numeric_features",
            pp.DropNonNumericFeatures(),
        ),
        (
            "gb_model",
            LogisticRegression(),
        ),
    ]
)
