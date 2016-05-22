import numpy as np

from uplift.ensemble import UpliftRandomForestClassifier

# toy_sample
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 1, 0, 1])

group = np.array([0, 0, 1, 1])

def test_forest_fits_predicts():
    urfc = UpliftRandomForestClassifier()
    urfc.fit(X, y, group)
    d_hat = urfc.predict_proba(X)

