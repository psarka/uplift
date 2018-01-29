import numpy as np

from uplift.validation.check import check_random_state


def make_radcliffe_surry(n_samples=64000, p_target=0.5,  a_positive=0.5, target_base_uplift=0.03,
                         target_b_uplift=0.1, return_uplift=False, random_state=None):

    generator = check_random_state(random_state)
    nrm = a_positive + target_base_uplift + target_b_uplift

    X = generator.rand(n_samples, 2)
    group = generator.binomial(1, p_target, size=n_samples)

    p_control = X[:, 0] * a_positive / nrm
    uplift = (X[:, 1]*target_b_uplift + target_base_uplift) / nrm

    y = generator.binomial(1, p_control + uplift * group)

    if return_uplift:
        return X, y, group, uplift
    else:
        return X, y, group


def negative_effects(n_samples=64000, p_target=0.5, a_positive=0.5, target_base_uplift=0.03,
                     target_b_uplift=0.1, return_uplift=False, random_state=None):

    generator = check_random_state(random_state)
    nrm = a_positive + target_base_uplift + target_b_uplift / 2.0

    X = generator.rand(n_samples, 2)
    group = generator.binomial(1, p_target, size=n_samples)

    p_control = X[:, 0] * a_positive / nrm
    uplift = ((X[:, 1] - 0.5)*target_b_uplift + target_base_uplift) / nrm

    y = generator.binomial(1, np.maximum(0, p_control + uplift * group))

    if return_uplift:
        return X, y, group, uplift
    else:
        return X, y, group
