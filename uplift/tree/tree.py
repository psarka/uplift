import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.tree import BaseDecisionTree, CRITERIA_CLF
from ._criterion import UpliftEntropy, ModifiedUpliftEntropy

CRITERIA_CLF['uplift_entropy'] = UpliftEntropy
CRITERIA_CLF['modified_entropy'] = ModifiedUpliftEntropy


class UpliftDecisionTreeClassifier(DecisionTreeClassifier, BaseDecisionTree):

    def __init__(self,
                 criterion="uplift_entropy",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 class_weight=None,
                 presort=False):
        super(UpliftDecisionTreeClassifier, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            presort=presort)

    def predict_proba(self, X, check_input=True):
        X = self._validate_X_predict(X, check_input)
        proba = self.tree_.predict(X)

        control_0 = proba[:, 0]
        control_1 = proba[:, 1]
        target_0 = proba[:, 2]
        target_1 = proba[:, 3]

        control = control_0 + control_1
        target = target_0 + target_1

        p_control = np.where(control == 0, np.zeros_like(control), control_1 / control)
        p_target = np.where(target == 0, np.zeros_like(target), target_1 / target)

        return p_target - p_control

