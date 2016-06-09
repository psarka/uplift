"""
The :mod:`sklearn.ensemble` module includes ensemble-based methods for
classification, regression and anomaly detection.
"""

from .base import BaseEnsemble
from .forest import RandomForestClassifier
from .forest import RandomForestRegressor
from .forest import RandomTreesEmbedding
from .forest import ExtraTreesClassifier
from .forest import ExtraTreesRegressor
from .bagging import BaggingClassifier
from .bagging import BaggingRegressor
from .gradient_boosting import GradientBoostingClassifier
from .gradient_boosting import GradientBoostingRegressor
from .voting_classifier import VotingClassifier

from . import bagging
from . import forest
from . import gradient_boosting
from . import partial_dependence

__all__ = ["BaseEnsemble",
           "RandomForestClassifier", "RandomForestRegressor",
           "RandomTreesEmbedding", "ExtraTreesClassifier",
           "ExtraTreesRegressor", "BaggingClassifier",
           "BaggingRegressor", "GradientBoostingClassifier",
           "GradientBoostingRegressor", "VotingClassifier",
           "bagging", "forest", "gradient_boosting",
           "partial_dependence"]
