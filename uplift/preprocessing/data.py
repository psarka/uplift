# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Andreas Mueller <amueller@ais.uni-bonn.de>
#          Eric Martin <eric@ericmart.in>
#          Giorgio Patrini <giorgio.patrini@anu.edu.au>
# License: BSD 3 clause

import numbers

import numpy as np
from scipy import sparse

from uplift.base import BaseEstimator
from uplift.base import TransformerMixin
from uplift.validation.check import check_array

FLOAT_DTYPES = (np.float64, np.float32, np.float16)


def _transform_selected(X, transform, selected="all", copy=True):
    """Apply a transform function to portion of selected features

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
        Dense array or sparse matrix.

    transform : callable
        A callable transform(X) -> X_transformed

    copy : boolean, optional
        Copy X even if it could be avoided.

    selected: "all" or array of indices or mask
        Specify which features to apply the transform to.

    Returns
    -------
    X : array or sparse matrix, shape=(n_samples, n_features_new)
    """
    if isinstance(selected, str) and selected == "all":
        return transform(X)

    X = check_array(X, accept_sparse='csc', copy=copy, dtype=FLOAT_DTYPES)

    if len(selected) == 0:
        return X

    n_features = X.shape[1]
    ind = np.arange(n_features)
    sel = np.zeros(n_features, dtype=bool)
    sel[np.asarray(selected)] = True
    not_sel = np.logical_not(sel)
    n_selected = np.sum(sel)

    if n_selected == 0:
        # No features selected.
        return X
    elif n_selected == n_features:
        # All features selected.
        return transform(X)
    else:
        X_sel = transform(X[:, ind[sel]])
        X_not_sel = X[:, ind[not_sel]]

        if sparse.issparse(X_sel) or sparse.issparse(X_not_sel):
            return sparse.hstack((X_sel, X_not_sel))
        else:
            return np.hstack((X_sel, X_not_sel))


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical integer features using a one-hot aka one-of-K scheme.

    The input to this transformer should be a matrix of integers, denoting
    the values taken on by categorical (discrete) features. The output will be
    a sparse matrix where each column corresponds to one possible value of one
    feature. It is assumed that input features take on values in the range
    [0, n_values).

    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.

    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.

    Parameters
    ----------
    n_values : 'auto', int or array of ints
        Number of values per feature.

        - 'auto' : determine value range from training data.
        - int : number of categorical values per feature.
                Each feature value should be in ``range(n_values)``
        - array : ``n_values[i]`` is the number of categorical values in
                  ``X[:, i]``. Each feature value should be in ``range(n_values[i])``

    categorical_features: "all" or array of indices or mask
        Specify what features are treated as categorical.

        - 'all' (default): All features are treated as categorical.
        - array of indices: Array of categorical feature indices.
        - mask: Array of length n_features and with dtype=bool.

        Non-categorical features are always stacked to the right of the matrix.

    dtype : number type, default=np.float
        Desired dtype of output.

    sparse : boolean, default=True
        Will return sparse matrix if set True else will return an array.

    handle_unknown : str, 'error' or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform.

    Attributes
    ----------
    active_features_ : array
        Indices for active features, meaning values that actually occur
        in the training set. Only available when n_values is ``'auto'``.

    feature_indices_ : array of shape (n_features,)
        Indices to feature ranges.
        Feature ``i`` in the original data is mapped to features
        from ``feature_indices_[i]`` to ``feature_indices_[i+1]``
        (and then potentially masked by `active_features_` afterwards)

    n_values_ : array of shape (n_features,)
        Maximum number of values per feature.

    Examples
    --------
    Given a dataset with three features and two samples, we let the encoder
    find the maximum value per feature and transform the data to a binary
    one-hot encoding.

    See also
    --------
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """
    def __init__(self, n_values="auto", categorical_features="all",
                 dtype=np.float64, sparse=True, handle_unknown='error'):
        self.n_values = n_values
        self.categorical_features = categorical_features
        self.dtype = dtype
        self.sparse = sparse
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit OneHotEncoder to X.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
            Input array of type int.

        Returns
        -------
        self
        """
        self.fit_transform(X)
        return self

    def _fit_transform(self, X):
        """Assumes X contains only categorical features."""
        X = check_array(X, dtype=np.int)
        if np.any(X < 0):
            raise ValueError("X needs to contain only non-negative integers.")
        n_samples, n_features = X.shape
        if (isinstance(self.n_values, str) and
            self.n_values == 'auto'):
            n_values = np.max(X, axis=0) + 1
        elif isinstance(self.n_values, numbers.Integral):
            if (np.max(X, axis=0) >= self.n_values).any():
                raise ValueError("Feature out of bounds for n_values=%d"
                                 % self.n_values)
            n_values = np.empty(n_features, dtype=np.int)
            n_values.fill(self.n_values)
        else:
            try:
                n_values = np.asarray(self.n_values, dtype=int)
            except (ValueError, TypeError):
                raise TypeError("Wrong type for parameter `n_values`. Expected"
                                " 'auto', int or array of ints, got %r"
                                % type(X))
            if n_values.ndim < 1 or n_values.shape[0] != X.shape[1]:
                raise ValueError("Shape mismatch: if n_values is an array,"
                                 " it has to be of shape (n_features,).")

        self.n_values_ = n_values
        n_values = np.hstack([[0], n_values])
        indices = np.cumsum(n_values)
        self.feature_indices_ = indices

        column_indices = (X + indices[:-1]).ravel()
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)
        data = np.ones(n_samples * n_features)
        out = sparse.coo_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()

        if (isinstance(self.n_values, str) and
            self.n_values == 'auto'):
            mask = np.array(out.sum(axis=0)).ravel() != 0
            active_features = np.where(mask)[0]
            out = out[:, active_features]
            self.active_features_ = active_features

        return out if self.sparse else out.toarray()

    def fit_transform(self, X, y=None):
        """Fit OneHotEncoder to X, then transform X.

        Equivalent to self.fit(X).transform(X), but more convenient and more
        efficient. See fit for the parameters, transform for the return value.
        """
        return _transform_selected(X, self._fit_transform,
                                   self.categorical_features, copy=True)

    def _transform(self, X):
        """Assumes X contains only categorical features."""
        X = check_array(X, dtype=np.int)
        if np.any(X < 0):
            raise ValueError("X needs to contain only non-negative integers.")
        n_samples, n_features = X.shape

        indices = self.feature_indices_
        if n_features != indices.shape[0] - 1:
            raise ValueError("X has different shape than during fitting."
                             " Expected %d, got %d."
                             % (indices.shape[0] - 1, n_features))

        # We use only those categorical features of X that are known using fit.
        # i.e lesser than n_values_ using mask.
        # This means, if self.handle_unknown is "ignore", the row_indices and
        # col_indices corresponding to the unknown categorical feature are
        # ignored.
        mask = (X < self.n_values_).ravel()
        if np.any(~mask):
            if self.handle_unknown not in ['error', 'ignore']:
                raise ValueError("handle_unknown should be either error or "
                                 "unknown got %s" % self.handle_unknown)
            if self.handle_unknown == 'error':
                raise ValueError("unknown categorical feature present %s "
                                 "during transform." % X.ravel()[~mask])

        column_indices = (X + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(np.sum(mask))
        out = sparse.coo_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if (isinstance(self.n_values, str) and
            self.n_values == 'auto'):
            out = out[:, self.active_features_]

        return out if self.sparse else out.toarray()

    def transform(self, X):
        """Transform X using one-hot encoding.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Input array of type int.

        Returns
        -------
        X_out : sparse matrix if sparse=True else a 2-d array, dtype=int
            Transformed input.
        """
        return _transform_selected(X, self._transform,
                                   self.categorical_features, copy=True)
