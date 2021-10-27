from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from config import REPLACE_MISSING_VALUE, DROP_FEATURES, MAX_CARDINALITY
import preprocessing as pp

loan_pipe = Pipeline(
    [
        (
            "drop_user_specified_features",
            pp.DropFeatures(drop_features=DROP_FEATURES),
        ),
        (
            "convert_categorical_vars_to_ordinal",
            pp.ConvertCategoricalToOrdinal(max_cardinality=MAX_CARDINALITY),
        ),
        (
            "drop_non_numeric_features",
            pp.DropNonNumericFeatures(),
        ),
        (
            "mean_imputation",
            pp.MeanImputer(),
        ),
        (
            "gb_model",
            LogisticRegression(),
        ),
    ]
)
