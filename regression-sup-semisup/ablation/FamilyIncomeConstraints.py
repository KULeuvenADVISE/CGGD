"""Copyright (c) DTAI - KU Leuven - All right reserved.

Proprietary, do not copy or distribute without permission.

Written by Quinten Van Baelen ORCID iD 0000-0003-2863-4227, 2021."""

import tensorflow as tf  # requires tf >= 2.2

# define the constraints
# sum constraint food expense: Total Food Expenditure (0th output variable) is larger than (or equal) the sum of Bread and Cereals Expenditure (1st output variable), Meat Expenditure (2nd output variable) and Vegetables Expenditure (3rd output variable)
# sum constraint income: Total Household Income (0th input variable) is larger than (or equal) the sum of Total Food Expenditure (0th output variable), Housing and water Expenditure (4th output variable), Medical Care Expenditure (5th output variable), Communication Expenditure (6th output variable) and Education Expenditure (7th output variable)
# for all output variables, 2 bound constraints are used. They are the maximum and minimum available in the dataset adjusted with some margin to allow for some flexibility out of the dataset.


def make_list_bound_constraints(margin):
    """ For the poroelastic materials construct a list containing the bound constraints and the bounds used for normalizing the data.

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
    lower_bound_income : list[tf.Tensor]
        List of tensor with dtype=tf.float32 that is the lower bound used in the normalization of Total Household Income.
    upper_bound_income : list[tf.Tensor]
        List of tensor with dtype=tf.float32 that is the upper bound used in the normalization of Total Household Income.

    """
    # output variables are normalized to [0, 1]
    list_bound_constraints = [[tf.constant(0, dtype=tf.int32), tf.subtract(tf.constant(0, dtype=tf.float32), margin),
                               tf.constant('larger', dtype=tf.string), tf.constant('equal', dtype=tf.string)],
                              [tf.constant(0, dtype=tf.int32), tf.add(tf.constant(1, dtype=tf.float32), margin),
                               tf.constant('smaller', dtype=tf.string), tf.constant('equal', dtype=tf.string)],
                              [tf.constant(1, dtype=tf.int32), tf.subtract(tf.constant(0, dtype=tf.float32), margin),
                               tf.constant('larger', dtype=tf.string), tf.constant('equal', dtype=tf.string)],
                              [tf.constant(1, dtype=tf.int32), tf.add(tf.constant(1, dtype=tf.float32), margin),
                               tf.constant('smaller', dtype=tf.string), tf.constant('equal', dtype=tf.string)],
                              [tf.constant(2, dtype=tf.int32), tf.subtract(tf.constant(0, dtype=tf.float32), margin),
                               tf.constant('larger', dtype=tf.string), tf.constant('equal', dtype=tf.string)],
                              [tf.constant(2, dtype=tf.int32), tf.add(tf.constant(1, dtype=tf.float32), margin),
                               tf.constant('smaller', dtype=tf.string), tf.constant('equal', dtype=tf.string)],
                              [tf.constant(3, dtype=tf.int32), tf.subtract(tf.constant(0, dtype=tf.float32), margin),
                               tf.constant('larger', dtype=tf.string), tf.constant('equal', dtype=tf.string)],
                              [tf.constant(3, dtype=tf.int32), tf.add(tf.constant(1, dtype=tf.float32), margin),
                               tf.constant('smaller', dtype=tf.string), tf.constant('equal', dtype=tf.string)],
                              [tf.constant(4, dtype=tf.int32), tf.subtract(tf.constant(0, dtype=tf.float32), margin),
                               tf.constant('larger', dtype=tf.string), tf.constant('equal', dtype=tf.string)],
                              [tf.constant(4, dtype=tf.int32), tf.add(tf.constant(1, dtype=tf.float32), margin),
                               tf.constant('smaller', dtype=tf.string), tf.constant('equal', dtype=tf.string)],
                              [tf.constant(5, dtype=tf.int32), tf.subtract(tf.constant(0, dtype=tf.float32), margin),
                               tf.constant('larger', dtype=tf.string), tf.constant('equal', dtype=tf.string)],
                              [tf.constant(5, dtype=tf.int32), tf.add(tf.constant(1, dtype=tf.float32), margin),
                               tf.constant('smaller', dtype=tf.string), tf.constant('equal', dtype=tf.string)],
                              [tf.constant(6, dtype=tf.int32), tf.subtract(tf.constant(0, dtype=tf.float32), margin),
                               tf.constant('larger', dtype=tf.string), tf.constant('equal', dtype=tf.string)],
                              [tf.constant(6, dtype=tf.int32), tf.add(tf.constant(1, dtype=tf.float32), margin),
                               tf.constant('smaller', dtype=tf.string), tf.constant('equal', dtype=tf.string)],
                              [tf.constant(7, dtype=tf.int32), tf.subtract(tf.constant(0, dtype=tf.float32), margin),
                               tf.constant('larger', dtype=tf.string), tf.constant('equal', dtype=tf.string)],
                              [tf.constant(7, dtype=tf.int32), tf.add(tf.constant(1, dtype=tf.float32), margin),
                               tf.constant('smaller', dtype=tf.string), tf.constant('equal', dtype=tf.string)]]

    lower_bound_unnormalized = [tf.constant(3704, dtype=tf.float32), tf.constant(0, dtype=tf.float32),
                                tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32),
                                tf.constant(1950, dtype=tf.float32), tf.constant(0, dtype=tf.float32),
                                tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32)]
    upper_bound_unnormalized = [tf.constant(791848, dtype=tf.float32), tf.constant(437467, dtype=tf.float32),
                                tf.constant(140992, dtype=tf.float32), tf.constant(74800, dtype=tf.float32),
                                tf.constant(2188560, dtype=tf.float32), tf.constant(1049275, dtype=tf.float32),
                                tf.constant(149940, dtype=tf.float32), tf.constant(731000, dtype=tf.float32)]

    lower_bound_income = tf.constant(11285, dtype=tf.float32)
    upper_bound_income = tf.constant(11815988, dtype=tf.float32)

    return list_bound_constraints, lower_bound_unnormalized, upper_bound_unnormalized, lower_bound_income, upper_bound_income


