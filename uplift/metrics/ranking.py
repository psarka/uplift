import numpy as np


def _uplift(responses_control, responses_target, n_control, n_target):
    if n_control == 0:
        return responses_target
    else:
        return responses_target - responses_control * n_target / n_control


def uplift_curve(y_true, d_pred, group, n_nodes=None):
    if n_nodes is None:
        n_nodes = min(len(y_true) + 1, 201)

    sorted_ds = sorted(zip(d_pred, group, y_true), reverse=True)
    responses_control, responses_target, n_control, n_target = 0, 0, 0, 0
    cumulative_responses = [(responses_control, responses_target, n_control, n_target)]

    for _, is_target, response in sorted_ds:
        if is_target:
            n_target += 1
            responses_target += response
        else:
            n_control += 1
            responses_control += response
        cumulative_responses.append((responses_control, responses_target, n_control, n_target))

    xs = [int(i) for i in np.linspace(0, len(y_true), n_nodes)]
    ys = [_uplift(*cumulative_responses[x]) for x in xs]

    return xs, ys


def number_responses(y_true, group):

    responses_target, responses_control, n_target, n_control = 0, 0, 0, 0
    for is_target, y in zip(group, y_true):
        if is_target:
            n_target += 1
            responses_target += y
        else:
            n_control += 1
            responses_control += y

    rescaled_responses_control = 0 if n_control == 0 else responses_control * n_target / n_control

    return responses_target, rescaled_responses_control


def optimal_uplift_curve(y_true, group, negative_effects=True):

    responses_target, rescaled_responses_control = number_responses(y_true, group)

    if negative_effects:
        xs = [0, responses_target, len(y_true) - rescaled_responses_control, len(y_true)]
        ys = [0, responses_target, responses_target, responses_target - rescaled_responses_control]
    else:
        xs = [0, responses_target - rescaled_responses_control, len(y_true)]
        ys = [0, responses_target - rescaled_responses_control, responses_target - rescaled_responses_control]

    return xs, ys


def null_uplift_curve(y_true, group):

    responses_target, rescaled_responses_control = number_responses(y_true, group)
    return [0, len(y_true)], [0, responses_target - rescaled_responses_control]


def area_under_curve(xs, ys):
    area = 0
    for i in range(1, len(xs)):
        delta = xs[i] - xs[i-1]
        y = (ys[i] + ys[i-1])/2
        area += y*delta
    return area


def qini(y_true, d_pred, group, negative_effects):

    area_optimal = area_under_curve(*optimal_uplift_curve(y_true, group, negative_effects))
    area_model = area_under_curve(*uplift_curve(y_true, d_pred, group))
    area_null = area_under_curve(*null_uplift_curve(y_true, group))

    return (area_model - area_null) / (area_optimal - area_null)


def qini_Q(y_true, d_pred, group):

    return qini(y_true, d_pred, group, negative_effects=True)


def qini_q(y_true, d_pred, group):

    return qini(y_true, d_pred, group, negative_effects=False)
