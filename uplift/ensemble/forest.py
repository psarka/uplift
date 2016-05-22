from sklearn.ensemble.forest import ForestClassifier
from ..tree import UpliftDecisionTreeClassifier


class UpliftRandomForestClassifier(ForestClassifier):

    def __init__(self,
                 n_estimators=10,
                 criterion='uplift_entropy',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features='auto',
                 max_leaf_nodes=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(UpliftRandomForestClassifier, self).__init__(
            base_estimator=UpliftDecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=('criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf',
                              'min_weight_fraction_leaf', 'max_features', 'max_leaf_nodes',
                              'random_state'),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes

    def fit(self, X, y, group):

        encoded_y = y + 2*group
        super(UpliftRandomForestClassifier, self).fit(X, encoded_y)
