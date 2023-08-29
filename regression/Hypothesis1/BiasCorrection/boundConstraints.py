"""

:author: Quinten Van Baelen (2022)
:license: Apache License, Version 2.0, see LICENSE for details.

"""

import tensorflow as tf  # requires tf >= 2.2


def make_list_bound_constraints(margin):
    """ For the bias correction dataset construct a list containing the bound constraints and the bounds used for normalizing the data.

    Parameters
    ----------
    margin : tf.Tensor
        Tensor indicating the margin put on the maxima and minima of the output variables in the dataset. This tensor should have
        dtype=tf.float32.

    Returns
    ----------
    list_bound_constraints : list[list[tf.Tensor]]
        List of list of tensors. The sublist represent a single bound constraint. The sublist has as its first element the index of the
        output variable (dtype=tf.int32) on which the bound constraint acts. The second element is the bound of the constraint
        (dtype=tf.float32). The third element is a string indicating if the bound is a smaller or larger constraint (dtype=tf.string).
        The fourth element is a string indicating if the constraint is strict or not (dtype=tf.string).
    lower_bound_unnormalized : list[tf.Tensor]
        List containing tensors (dtype=tf.float32) giving the lower bound that was used to normalize the ground truth of the output
        variables in the dataset.
    upper_bound_unnormalized : list[tf.Tensor]
        List containing tensors (dtype=tf.float32) giving the upper bound that was used to normalize the ground truth of the output
        variables in the dataset.

    """
    list_bound_constraints = [[tf.constant(0, dtype=tf.int32), tf.subtract(tf.constant(0, dtype=tf.float32), margin),
                               tf.constant('larger', dtype=tf.string), tf.constant('equal', dtype=tf.string)],
                              [tf.constant(0, dtype=tf.int32), tf.add(tf.constant(1, dtype=tf.float32), margin),
                               tf.constant('smaller', dtype=tf.string), tf.constant('equal', dtype=tf.string)],
                              [tf.constant(1, dtype=tf.int32), tf.subtract(tf.constant(0, dtype=tf.float32), margin),
                               tf.constant('larger', dtype=tf.string), tf.constant('equal', dtype=tf.string)],
                              [tf.constant(1, dtype=tf.int32), tf.add(tf.constant(1, dtype=tf.float32), margin),
                               tf.constant('smaller', dtype=tf.string), tf.constant('equal', dtype=tf.string)]]

    lower_bound_unnormalized = [tf.constant(11.3, dtype=tf.float32), tf.constant(11.3, dtype=tf.float32)]
    upper_bound_unnormalized = [tf.constant(38.9, dtype=tf.float32), tf.constant(38.9, dtype=tf.float32)]

    return list_bound_constraints, lower_bound_unnormalized, upper_bound_unnormalized

