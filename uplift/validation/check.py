import warnings

import numbers
import numpy as np
import scipy.sparse as sp

from uplift.exceptions import DataConversionWarning, NotFittedError


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def check_array(array, accept_sparse=None, dtype="numeric", order=None,
                copy=False, force_all_finite=True, ensure_2d=True,
                allow_nd=False, ensure_min_samples=1, ensure_min_features=1,
                warn_on_dtype=False, estimator=None):
    """Input validation on an array, list, sparse matrix or similar.

    By default, the input is converted to an at least 2D numpy array.
    If the dtype of the array is object, attempt converting to float,
    raising on failure.

    Parameters
    ----------
    array : object
        Input object to check / convert.

    accept_sparse : string, list of string or None (default=None)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc.  None means that sparse matrix input will raise an error.
        If the input is sparse but not in the allowed format, it will be
        converted to the first listed format.

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.
        When order is None (default), then if copy=False, nothing is ensured
        about the memory layout of the output array; otherwise (copy=True)
        the memory layout of the returned array is kept as close as possible
        to the original array.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.

    ensure_2d : boolean (default=True)
        Whether to make X at least 2d.

    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.

    ensure_min_samples : int (default=1)
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.

    warn_on_dtype : boolean (default=False)
        Raise DataConversionWarning if the dtype of the input data structure
        does not match the requested dtype, causing a memory copy.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    X_converted : object
        The converted and validated X.
    """
    if isinstance(accept_sparse, str):
        accept_sparse = [accept_sparse]

    # store whether originally we wanted numeric dtype
    dtype_numeric = dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, 'kind'):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = np.float64
        else:
            dtype = None

    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no dtype conversion required
            dtype = None
        else:
            # dtype conversion required. Let's select the first element of the
            # list of accepted types.
            dtype = dtype[0]

    if estimator is not None:
        if isinstance(estimator, str):
            estimator_name = estimator
        else:
            estimator_name = estimator.__class__.__name__
    else:
        estimator_name = "Estimator"
    context = " by %s" % estimator_name if estimator is not None else ""

    if sp.issparse(array):
        array = _ensure_sparse_format(array, accept_sparse, dtype, copy,
                                      force_all_finite)
    else:
        array = np.array(array, dtype=dtype, order=order, copy=copy)

        if ensure_2d:
            if array.ndim == 1:
                if ensure_min_samples >= 2:
                    raise ValueError("%s expects at least 2 samples provided "
                                     "in a 2 dimensional array-like input"
                                     % estimator_name)
                warnings.warn(
                    "Passing 1d arrays as data is deprecated in 0.17 and will "
                    "raise ValueError in 0.19. Reshape your data either using "
                    "X.reshape(-1, 1) if your data has a single feature or "
                    "X.reshape(1, -1) if it contains a single sample.",
                    DeprecationWarning)
            array = np.atleast_2d(array)
            # To ensure that array flags are maintained
            array = np.array(array, dtype=dtype, order=order, copy=copy)

        # make sure we actually converted to numeric:
        if dtype_numeric and array.dtype.kind == "O":
            array = array.astype(np.float64)
        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. %s expected <= 2."
                             % (array.ndim, estimator_name))
        if force_all_finite:
            _assert_all_finite(array)

    shape_repr = _shape_repr(array.shape)
    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                             " minimum of %d is required%s."
                             % (n_samples, shape_repr, ensure_min_samples,
                                context))

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required%s."
                             % (n_features, shape_repr, ensure_min_features,
                                context))

    if warn_on_dtype and dtype_orig is not None and array.dtype != dtype_orig:
        msg = ("Data with input dtype %s was converted to %s%s."
               % (dtype_orig, array.dtype, context))
        warnings.warn(msg, DataConversionWarning)
    return array


def check_is_fitted(estimator, attributes, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    "all_or_any" of the passed attributes and raises a NotFittedError with the
    given message.

    Parameters
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.

    attributes : attribute name(s) given as string or a list/tuple of strings
        Eg. : ["coef_", "estimator_", ...], "coef_"

    msg : string
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this method."

        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.

        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

    all_or_any : callable, {all, any}, default all
        Specify whether all or any of the given attributes must exist.
    """
    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {'name': type(estimator).__name__})


def assert_all_finite(X):
    """Throw a ValueError if X contains NaN or infinity.

    Input MUST be an np.ndarray instance or a scipy.sparse matrix."""
    _assert_all_finite(X.data if sp.issparse(X) else X)


def _assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method.
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)


def _ensure_sparse_format(spmatrix, accept_sparse, dtype, copy,
                          force_all_finite):
    """Convert a sparse matrix to a given format.

    Checks the sparse format of spmatrix and converts if necessary.

    Parameters
    ----------
    spmatrix : scipy sparse matrix
        Input to validate and convert.

    accept_sparse : string, list of string or None (default=None)
        String[s] representing allowed sparse matrix formats ('csc',
        'csr', 'coo', 'dok', 'bsr', 'lil', 'dia'). None means that sparse
        matrix input will raise an error.  If the input is sparse but not in
        the allowed format, it will be converted to the first listed format.

    dtype : string, type or None (default=none)
        Data type of result. If None, the dtype of the input is preserved.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.

    Returns
    -------
    spmatrix_converted : scipy sparse matrix.
        Matrix that is ensured to have an allowed type.
    """
    if accept_sparse in [None, False]:
        raise TypeError('A sparse matrix was passed, but dense '
                        'data is required. Use X.toarray() to '
                        'convert to a dense numpy array.')
    if dtype is None:
        dtype = spmatrix.dtype

    changed_format = False
    if (isinstance(accept_sparse, (list, tuple))
            and spmatrix.format not in accept_sparse):
        # create new with correct sparse
        spmatrix = spmatrix.asformat(accept_sparse[0])
        changed_format = True

    if dtype != spmatrix.dtype:
        # convert dtype
        spmatrix = spmatrix.astype(dtype)
    elif copy and not changed_format:
        # force copy
        spmatrix = spmatrix.copy()

    if force_all_finite:
        if not hasattr(spmatrix, "data"):
            warnings.warn("Can't check %s sparse matrix for nan or inf."
                          % spmatrix.format)
        else:
            _assert_all_finite(spmatrix.data)
    return spmatrix


def _shape_repr(shape):
    """Return a platform independent representation of an array shape

    Under Python 2, the `long` type introduces an 'L' suffix when using the
    default %r format for tuples of integers (typically used to store the shape
    of an array).

    Under Windows 64 bit (and Python 2), the `long` type is used by default
    in numpy shapes even when the integer dimensions are well below 32 bit.
    The platform specific type causes string messages or doctests to change
    from one platform to another which is not desirable.

    Under Python 3, there is no more `long` type so the `L` suffix is never
    introduced in string representation.

    >>> _shape_repr((1, 2))
    '(1, 2)'
    >>> one = 2 ** 64 / 2 ** 64  # force an upcast to `long` under Python 2
    >>> _shape_repr((one, 2 * one))
    '(1, 2)'
    >>> _shape_repr((1,))
    '(1,)'
    >>> _shape_repr(())
    '()'
    """
    if len(shape) == 0:
        return "()"
    joined = ", ".join("%d" % e for e in shape)
    if len(shape) == 1:
        # special notation for singleton tuples
        joined += ','
    return "(%s)" % joined


def _num_samples(x):
    """Return number of samples in array-like x."""
    if hasattr(x, 'fit'):
        # Don't get num_samples from an ensembles length!
        raise TypeError('Expected sequence or array-like, got '
                        'estimator %s' % x)
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)


def column_or_1d(y, warn=False):
    """ Ravel column or 1d numpy array, else raises an error
    Parameters
    ----------
    y : array-like
    warn : boolean, default False
       To control display of warnings.
    Returns
    -------
    y : array
    """
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn("A column-vector y was passed when a 1d array was"
                          " expected. Please change the shape of y to "
                          "(n_samples, ), for example using ravel().",
                          DataConversionWarning, stacklevel=2)
        return np.ravel(y)

    raise ValueError("bad input shape {0}".format(shape))

