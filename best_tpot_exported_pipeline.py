import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=32)

# Average CV score on the training set was: 0.9554195394212279
exported_pipeline = make_pipeline(
    make_union(
        make_union(
            FunctionTransformer(copy),
            make_union(
                FunctionTransformer(copy),
                FunctionTransformer(copy)
            )
        ),
        FunctionTransformer(copy)
    ),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=7, max_features=1.0, min_samples_leaf=6, min_samples_split=9, n_estimators=100, subsample=0.9500000000000001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 32)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
