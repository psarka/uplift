import numpy as np

from uplift.tree import UpliftDecisionTreeClassifier

# toy_sample
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 1, 0, 1])

group = np.array([0, 0, 1, 1])

def test_tree_fits_predicts():
     udtc = UpliftDecisionTreeClassifier()
     udtc.fit(X, 2*group + y)
     d_hat = udtc.predict_proba(X)
