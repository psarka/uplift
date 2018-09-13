"""
Tests for the uplift.forest module
"""

# Authors: Paulius Šarka
# License: BSD 3 clause

import pytest

from uplift.ensemble import RandomForestClassifier
from uplift.datasets import make_radcliffe_surry
from uplift.metrics import qini_q


def test_example():

    X_train, y_train, group_train = make_radcliffe_surry(n_samples=1000, random_state=1)
    X_test, y_test, group_test = make_radcliffe_surry(n_samples=1000, random_state=2)

    rfc = RandomForestClassifier(n_estimators=50,
                                 min_samples_leaf=200,
                                 criterion='uplift_gini',
                                 random_state=1)

    rfc.fit(X_train, y_train, group_train)
    uplift_pred = rfc.predict_uplift(X_test)

    assert qini_q(y_test, uplift_pred, group_test) == pytest.approx(0.261, abs=0.01)
