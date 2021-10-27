from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from config import REPLACE_MISSING_VALUE, DROP_FEATURES, MAX_CARDINALITY
import preprocessing as pp

class SklearnPipeline:
    """Class to construct an Sklearn pipeline."""
    def __init__(self, 
                 convert_categorical_to_ordinal=True,
                 mean_imputation=True,
                 algorithm=LogisticRegression()
                ):

        self.convert_categorical_to_ordinal = convert_categorical_to_ordinal
        self.mean_imputation = mean_imputation
        self.algorithm = algorithm

    def make_pipeline(self):
        """Make the pipeline."""

        pipe_list = [(
                    "drop_user_specified_features",
                    pp.DropFeatures(drop_features=DROP_FEATURES),
                )]

        if self.convert_categorical_to_ordinal:
            pipe_list.append((
                             "convert_categorical_vars_to_ordinal",
                             pp.ConvertCategoricalToOrdinal(max_cardinality=MAX_CARDINALITY),
                            ))

        # Always drop the non-numeric features!
        pipe_list.append((
                          "drop_non_numeric_features",
                          pp.DropNonNumericFeatures(),
                         ))

        if self.mean_imputation:
            pipe_list.append((
                              "mean_imputation",
                              pp.MeanImputer(),
                            ))
                            
        # Add the model to the pipeline
        pipe_list.append((
                          "model",
                          self.algorithm,
                         ))

        return Pipeline(pipe_list,)
        