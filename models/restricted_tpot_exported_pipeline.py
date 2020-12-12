import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=32)

# Average CV score on the training set was: 0.932505472401108
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=BernoulliNB(alpha=100.0, fit_prior=False)),
    RFE(estimator=ExtraTreesClassifier(criterion="entropy", max_features=0.8, n_estimators=100), step=0.9000000000000001),
    Normalizer(norm="l1"),
    GradientBoostingClassifier(learning_rate=0.01, max_depth=2, max_features=0.9500000000000001, min_samples_leaf=7, min_samples_split=19, n_estimators=100, subsample=0.3)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 32)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
