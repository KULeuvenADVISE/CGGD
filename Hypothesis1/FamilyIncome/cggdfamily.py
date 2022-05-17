"""

:author: Quinten Van Baelen (2022)
:license: Apache License, Version 2.0, see LICENSE for details.

"""

import numpy as np
import tensorflow as tf  # requires tf >=2.2
import os
import time


def get_neur_act_seq(model, input_dimension):
    """Get list with integers representing the first dimension of the weight matrices and the number of activation functions.

    Parameters
    ----------
    model : tf.keras.Sequential
        model that is trained defined as a Sequential model
    input_dimension : int
        integer denoting the dimensionality of the input of the network

    Returns
    ----------
    neur_act : list[int]
        list with integers representing the first dimension of the weight matrices and the number of activation functions

    It is assumed that the model is dense. This allows for a very efficient and easy way to compute the dimension of the weight matrices
    and the number of activation functions. Namely the dimension of the weight matrices is determined by the number of neurons in that
    layer and the next layer. The number of activation functions is the number of neurons in that layer.

    This method works (or should work, still needs testing) even for dense networks where the output shape in some layers is higher
    dimensional.
    """
    num_layers = len(model.layers)
    neur_act = [input_dimension]
    for layer_index in range(0, num_layers):
        layer = model.layers[layer_index]
        if layer_index < num_layers-1:
            for counter_dim in range(0, 2):
                for counter_shape in range(1, len(layer.output.shape)):
                    neur_act.append(int(layer.output.shape[counter_shape]))
        else:
            for counter_shape in range(1, len(layer.output.shape)):
                neur_act.append(int(layer.output.shape[counter_shape]))
    return neur_act


def get_neur_act_fun(model):
    """Get list with integers representing the first dimension of the weight matrices and the number of activation functions.

    Parameters
    ----------
    model : tf.keras.Model
        model that is trained defined as a model using the functional API

    Returns
    ----------
    neur_act : list[int]
        list with integers representing the first dimension of the weight matrices and the number of activation functions

    It is assumed that the model is dense. This allows for a very efficient and easy way to compute the dimension of the weight matrices
    and the number of activation functions. Namely the dimension of the weight matrices is determined by the number of neurons in that
    layer and the next layer. The number of activation functions is the number of neurons in that layer.

    This method works (or should work, still needs testing) even for dense networks where the output shape in some layers is higher
    dimensional.
    """
    num_layers = len(model.layers)
    neur_act = []
    for layer_index in range(0, num_layers):
        layer = model.layers[layer_index]
        if 0 < layer_index < num_layers-1:
            for counter_dim in range(0, 2):
                for counter_shape in range(1, len(layer.output.shape)):
                    neur_act.append(int(layer.output.shape[counter_shape]))
        else:
            for counter_shape in range(1, len(layer.output.shape)):
                neur_act.append(int(layer.output.shape[counter_shape]))
    return neur_act


def get_neur_act(model, input_dimension):
    """Get list with integers representing the first dimension of the weight matrices and the number of activation functions.

    It is assumed that the model is dense. This allows for a very efficient and easy way to compute the dimension of the weight matrices
    and the number of activation functions. Namely the dimension of the weight matrices is determined by the number of neurons in that
    layer and the next layer. The number of activation functions is the number of neurons in that layer.

    The difference between a Sequential and functional API model is needed due to the small difference there is between how the first layer
    is build out of the output. One can add here additional elif statements if one has a model of a different type. Note that one should
    write in this case a new get_number_neurons_new_model_type that computes the corresponding list of integers.

    Parameters
    ----------
    model : tf.keras.Model or tf.keras.Sequential
        model that is trained defined as a Sequential model or using the functional API
    input_dimension : int
        integer denoting the dimension of the input of the network

    Returns
    ----------
    neur_act : list[int]
        list with integers representing the first dimension of the weight matrices and the number of activation functions. If the model
        type is not recognized, then it returns the list [0].

    """
    if isinstance(model, tf.keras.Sequential):
        neur_act = get_neur_act_seq(model, input_dimension)
        print('Sequential model type.')
    elif isinstance(model, tf.keras.Model):
        neur_act = get_neur_act_fun(model)
        print('Functional API model type.')
    else:
        print('Invalid model type.')
        neur_act = [0]
    return neur_act


def convert_bound_con(index_output, value_bound, batch_size, output_dim):
    """Convert a constraint into a tf.tensor and scale according to the batch-size used in the training.

    The tensor that is returned consists mostly out of zeros except for the column with index convert_number_output. This column consists of
    convert_value_bound at each row. The result is used for checking if the output of the network satisfies this constraint by simply
    subtracting or adding this tensor from the predictions. Moreover, all the constraints corresponding to smaller or equal,
    strictly smaller, larger or equal and strictly larger are grouped in 4 different tensors. This allows for a simple elementwise
    computation of these four constant tensors and the output of the network.

    In order to construct the returned tensor, it is useful to make a case distinction on the index of the constrained output neuron.
    More specific, it is checked first that the constrained output neuron is the 0th output neuron, if not then it is checked that it is
    the last output neuron and if not we know that it is a neuron somewhere in the middle. This is necessary due to the stacking of the
    column vector with the bounds together with tensors consisting of only zeros.

    Currently, negative indexing is not supported. Note that if the convention of defining the constraints is followed, then negative
    indices should not be used.

    Parameters
    ----------
    index_output : tf.Tensor
        index of the output of the network that has the bound constraint. The standard Python indexing is used (e.g. the 0th element
        denotes the first element). The input has dtype=tf.float32 and shape=(,)
    value_bound : tf.Tensor
        value of the bound of the constraint. The input has dtype=tf.float32 and shape=(,)
    batch_size : int
        batch-size used in training
    output_dim : int
        the number of outputs of the network

    Returns
    ----------
    bound_constraint : tf.Tensor
        constant tensor containing the bound for each training example for a given bound constraint. This is a tensor containing 0
        except for the column corresponding to the constrained output variable. The input has dtype=tf.float32 shape=(batch_size,
        output_dim).
    indicator_constraint : tf.Tensor
        constant tensor indicating the outputs having a constraint. This is a tensor containing 0 except for the column corresponding
        to the constrained output variable. The input has dtype=tf.float32 shape=(batch_size, output_dim).

    """
    if index_output == tf.constant(0, dtype=tf.int32):
        # the constraint is on the 0th output neuron
        constant_part = tf.ones([batch_size, 1], dtype=tf.float32)
        zero_part = tf.zeros([batch_size, tf.math.subtract(output_dim, tf.constant(1, dtype=tf.int32))], dtype=tf.float32)
        indicator_constraint = tf.concat([constant_part, zero_part], axis=1)
    elif index_output == tf.math.subtract(output_dim, tf.constant(1, dtype=tf.int32)):
        # the constraint is on the last output neuron
        constant_part = tf.ones([batch_size, 1], dtype=tf.float32)
        zero_part = tf.zeros([batch_size, tf.math.subtract(output_dim, tf.constant(1, dtype=tf.int32))], dtype=tf.float32)
        indicator_constraint = tf.concat([zero_part, constant_part], axis=1)
    else:
        # the constraint is on some neuron in the middle
        constant_part = tf.ones([batch_size, 1], dtype=tf.float32)
        zero_part_first = tf.zeros([batch_size, index_output], dtype=tf.float32)
        rem_dimensions = tf.math.subtract(output_dim, tf.math.add(index_output, tf.constant(1, dtype=tf.int32)))
        zero_part_last = tf.zeros([batch_size, rem_dimensions], dtype=tf.float32)
        indicator_constraint = tf.concat([zero_part_first, constant_part, zero_part_last], axis=1)

    bound_constraint = tf.math.scalar_mul(value_bound, indicator_constraint)  # rescale the indicator tensor by the value of the bound
    return bound_constraint, indicator_constraint


def construct_bound_con(bound_constraints, batch_size, output_dim):
    """Construct a single tensor containing all the bound constraints on the output of the network.

    This function uses tf.function convert_bound_con. For more info about this function, please read the documentation provided there.

    This function goes through all the constraints one by one and add it together in the corresponding tensors. It is assumed here that
    there are no duplicates in the list bound_constraints. Although the method will not fail when there are duplicates present in this
    list, it is advised to not have this happen. The consequence of having duplicates in the list is that not all constraints will be taken
    into account with the same weight.

    This function is implemented in such a way that one can easily add other kind of constraints by using additional conditional statements.
    This will not impact the procedure of the currently implemented constraints by the choice of implementation.

    Parameters
    ----------
    bound_constraints : list[list[tf.Tensor]]
        list of the bound constraints, where each bound constraint consists of 4 tensors: output index, value of bound, smaller or larger
        string, and equal or strict string.
    batch_size : int
        integer representing the batch-size used during training.
    output_dim : int
        output dimension of the network

    Returns
    ----------
    bound_sm_eq : tf.Tensor
        consists of all the columns with bounds corresponding to smaller or equal constraints on the output. This tensor has
        dtype=tf.float32 and is of shape (batch_size, output_dim).
    bound_sm_st : tf.Tensor
        similarly defined as previous but for the strictly smaller constraints. This tensor has dtype=tf.float32 and is of shape
        (batch_size, output_dim).
    bound_la_eq : tf.Tensor
        similarly defined as previous but for the larger or equal constraints. This tensor has dtype=tf.float32 and is of shape
        (batch_size, output_dim).
    bound_la_st : tf.Tensor
        similarly defined as previous but for the strictly larger constraints. This tensor has dtype=tf.float32 and is of shape
        (batch_size, output_dim).
    ind_sm_eq : tf.Tensor
        corresponding indicator tensor of bound_sm_eq. This tensor has dtype=tf.float32 and is of shape (batch_size, output_dim).
    ind_sm_st : tf.Tensor
        similarly defined as ind_sm_eq but for the strictly smaller constraints. This tensor has dtype=tf.float32 and is of shape
        (batch_size, output_dim).
    ind_la_eq : tf.Tensor
        similarly defined as ind_sm_eq but for the larger or equal constraints. This tensor has dtype=tf.float32 and is of shape
        (batch_size, output_dim).
    ind_la_st : tf.Tensor
        similarly defined as ind_sm_eq but for the strictly larger constraints. This tensor has dtype=tf.float32 and is of shape
        (batch_size, output_dim).

    """
    # initialize each output vector
    bound_sm_eq = tf.zeros([batch_size, output_dim], dtype=tf.float32)
    bound_sm_st = tf.zeros([batch_size, output_dim], dtype=tf.float32)
    bound_la_eq = tf.zeros([batch_size, output_dim], dtype=tf.float32)
    bound_la_st = tf.zeros([batch_size, output_dim], dtype=tf.float32)

    ind_sm_eq = tf.zeros([batch_size, output_dim], dtype=tf.float32)
    ind_sm_st = tf.zeros([batch_size, output_dim], dtype=tf.float32)
    ind_la_eq = tf.zeros([batch_size, output_dim], dtype=tf.float32)
    ind_la_st = tf.zeros([batch_size, output_dim], dtype=tf.float32)

    # go over all constraints that were given
    for count_const in range(0, len(bound_constraints)):
        # check if constraint is of type smaller
        if bound_constraints[count_const][2] == tf.constant("smaller", dtype=tf.string):
            # check if constraint is of type equal
            if bound_constraints[count_const][3] == tf.constant("equal", dtype=tf.string):
                constant_part_smaller_equal, indicator_part_smaller_equal = \
                    convert_bound_con(bound_constraints[count_const][0],
                                      bound_constraints[count_const][1],
                                      tf.constant(batch_size, dtype=tf.int32),
                                      tf.constant(output_dim, dtype=tf.int32))
                bound_sm_eq = tf.math.add(bound_sm_eq, constant_part_smaller_equal)
                ind_sm_eq = tf.math.add(ind_sm_eq, indicator_part_smaller_equal)
            # check if constraint is of type strict
            elif bound_constraints[count_const][3] == tf.constant("strict", dtype=tf.string):
                constant_part_smaller_strict, indicator_part_smaller_strict = \
                    convert_bound_con(bound_constraints[count_const][0],
                                      bound_constraints[count_const][1],
                                      tf.constant(batch_size, dtype=tf.int32),
                                      tf.constant(output_dim, dtype=tf.int32))
                bound_sm_st = tf.math.add(bound_sm_st, constant_part_smaller_strict)
                ind_sm_st = tf.math.add(ind_sm_st, indicator_part_smaller_strict)
        # check if constraint is of type larger
        elif bound_constraints[count_const][2] == tf.constant("larger", dtype=tf.string):
            # check if constraint is of type equal
            if bound_constraints[count_const][3] == tf.constant("equal", dtype=tf.string):
                constant_part_larger_equal, indicator_part_larger_equal = \
                    convert_bound_con(bound_constraints[count_const][0],
                                      bound_constraints[count_const][1],
                                      tf.constant(batch_size, dtype=tf.int32),
                                      tf.constant(output_dim, dtype=tf.int32))
                bound_la_eq = tf.math.add(bound_la_eq, constant_part_larger_equal)
                ind_la_eq = tf.math.add(ind_la_eq, indicator_part_larger_equal)
            # check if constraint is of type strict
            elif bound_constraints[count_const][3] == tf.constant("strict", dtype=tf.string):
                constant_part_larger_strict, indicator_part_larger_strict = \
                    convert_bound_con(bound_constraints[count_const][0],
                                      bound_constraints[count_const][1],
                                      tf.constant(batch_size, dtype=tf.int32),
                                      tf.constant(output_dim, dtype=tf.int32))
                bound_la_st = tf.math.add(bound_la_st, constant_part_larger_strict)
                ind_la_st = tf.math.add(ind_la_st, indicator_part_larger_strict)

    return bound_sm_eq, bound_sm_st, bound_la_eq, bound_la_st, ind_sm_eq, ind_sm_st, ind_la_eq, ind_la_st


def determine_sat_con(outputs, bound_constraints, batch_size, output_dim, type_constraint, ind_sm_eq, ind_sm_st, ind_la_eq, ind_la_st):
    """Determine which constraints are satisfied and which constraints not.

    This function constructs a tensor consisting of ones and zeros. The tensor has a one at a certain position if the bound constraint
    corresponding to that output neuron and the output of the network is not satisfied. The tensor has a zero at that position if there is
    not constraint on the corresponding output or that the constraint is satisfied for the output of the network of this specific training
    example.

    If one does not give a valid type of bound constraint, that is, if type_constraint is not string Tensors equal to 'se', 'ss', 'le' or
    'sl', then the output will be the zero tensor to allow for further training but this means that the constraints are not taken into
    account during training. If this is the case, the sentence 'Invalid type of bound constraint.' is printed.

    This function is implemented in such a way that one can add additional type of constraints by adding additional conditions. This will
    not effect the procedure on the currently implemented kind of constraints.

    Parameters
    ----------
    outputs : tf.Tensor
        The outputs of the network of the current batch of training examples. This tensor has dtype=tf.float32 and
        shape=(batch_size, output_dim)
    bound_constraints : tf.Tensor
        constant tensor containing all the bound constraints. This tensor has dtype=tf.float32 and shape=(batch_size, output_dim)
    batch_size : int
        integer denoting the batch-size used in training.
    output_dim : int
        output dimension of the network
    type_constraint : tf.Tensor
        tensor denoting the type of the constraint, This tensor has dtype=tf.string.
    ind_sm_eq : tf.Tensor
        constant tensor indicating which output variables correspond to a smaller or equal constraint. This tensor is of dtype=tf.float32
        and of shape (batch_size, output_dim)
        corresponding indicator tensor of bound_sm_eq.
    ind_sm_st : tf.Tensor
        similarly defined as ind_sm_eq but for the strictly smaller constraints.
    ind_la_eq : tf.Tensor
        similarly defined as ind_sm_eq but for the larger or equal constraints.
    ind_la_st : tf.Tensor
        similarly defined as ind_sm_eq but for the strictly larger constraints.

    Returns
    ----------
    unsat_con : tf.Tensor
        Tensor indicating the satisfaction of the bound constraints for each training example in the batch. The value 1
        indicates that the constraint is not satisfied for that example, and the value 0 indicates that the constraint is satisfied. This
        tensor has dtype=tf.float32 and shape=(batch_size, output_dim).
    count_unsatisfied_con : tf.Tensor
        For the given outputs, the number of constraints that are not satisfied. This tensor has dtype=tf.float32 and shape=(,).
    count_total_con : tf.Tensor
        For the given outputs, the total number of variables that have a constrained. This is the same number as the batch-size times the
        length of the bound constraints. This tensor has dtype=tf.float32 and shape=(,).

    """
    correction = tf.ones_like(outputs)

    if type_constraint == tf.constant('se', dtype=tf.string):  # check if the bound constraints are smaller or equal
        correction = tf.math.subtract(correction, ind_sm_eq)
        proj_outputs = tf.math.multiply(outputs, ind_sm_eq)
        proj_outputs = tf.math.subtract(proj_outputs, correction)
        count_total_con = tf.cast(tf.math.reduce_sum(ind_sm_eq, axis=None), dtype=tf.float32)
        sat_con = tf.math.less_equal(proj_outputs, bound_constraints)
    elif type_constraint == tf.constant('ss', dtype=tf.string):  # check if the bound constraints are strictly smaller
        correction = tf.math.subtract(correction, ind_sm_st)
        proj_outputs = tf.math.multiply(outputs, ind_sm_st)
        proj_outputs = tf.math.subtract(proj_outputs, correction)  # the outputs that are unconstrained should be strictly less than 0
        count_total_con = tf.cast(tf.math.reduce_sum(ind_sm_st, axis=None), dtype=tf.float32)
        sat_con = tf.math.less(proj_outputs, bound_constraints)
    elif type_constraint == tf.constant('le', dtype=tf.string):  # check if the bound constraints are larger or equal
        correction = tf.math.subtract(correction, ind_la_eq)
        proj_outputs = tf.math.multiply(outputs, ind_la_eq)
        proj_outputs = tf.math.add(proj_outputs, correction)  # the outputs that are unconstrained should be strictly greater than 0
        count_total_con = tf.cast(tf.math.reduce_sum(ind_la_eq, axis=None), dtype=tf.float32)
        sat_con = tf.math.greater_equal(proj_outputs, bound_constraints)
    elif type_constraint == tf.constant('sl', dtype=tf.string):  # check if the bound constraints are strictly larger
        correction = tf.math.subtract(correction, ind_la_st)
        proj_outputs = tf.math.multiply(outputs, ind_la_st)
        proj_outputs = tf.math.add(proj_outputs, correction)  # the outputs that are unconstrained should be strictly greater than 0
        count_total_con = tf.cast(tf.math.reduce_sum(ind_la_st, axis=None), dtype=tf.float32)
        sat_con = tf.math.greater(proj_outputs, bound_constraints)
    else:
        sat_con = tf.constant(True, dtype=tf.bool, shape=[batch_size, output_dim])
        count_total_con = tf.constant(0, dtype=tf.float32)
        tf.print('Invalid type of bound constraint.')

    unsat_con = tf.where(sat_con, tf.zeros_like(bound_constraints), tf.ones_like(bound_constraints))

    count_unsat_con = tf.math.reduce_sum(unsat_con, axis=None)

    return unsat_con, count_unsat_con, count_total_con


def compute_sat_ratio(list_counter_unsatisfied, list_counter_total):
    """Compute the satisfaction rate of all constraints.

    This function computes the satisfaction ratio, which is defined as the number of satisfied constraints divided by the total number of
    constraints. Here both numbers are counted with respect to the training examples and not with respect the constraints themselves. For
    example, if there would be one constraint and the training is done over a batch of size 100. Then the satisfaction ratio is exactly the
    number of training examples that satisfy the constraint divided by 100.

    It is assumed that the two lists given as input have the same length.

    Parameters
    ----------
    list_counter_unsatisfied : list[tf.Tensor]
        list containing constant tensors denoting the number of unsatisfied constraints for a specific kind of constraint. These tensors
        are assumed to have dtype=tf.float32 and shape=(,) such that they can easily be used to compute the satisfaction ratio.
    list_counter_total : list[tf.Tensor]
        list containing constant tensors denoting the number of constraints for a specific kind of constraint. These tensors are assumed
        to have dtype=tf.float32 and shape=(,) such that they can easily be used to compute the satisfaction ratio.

    Returns
    ----------
    sat_ratio : tf.Tensor
        constant tensor indicating the percentage of constraints satisfied for the current batch. This tensor has dtype= tf.float32 and
        shape=(,).

    """
    total_unsatisfied = tf.constant(0, dtype=tf.float32)
    total = tf.constant(0, dtype=tf.float32)

    for i in range(0, len(list_counter_unsatisfied)):
        total_unsatisfied = tf.math.add(total_unsatisfied, list_counter_unsatisfied[i])
        total = tf.math.add(total, list_counter_total[i])

    total_satisfied = tf.math.subtract(total, total_unsatisfied)
    sat_ratio = tf.math.divide(total_satisfied, total)

    return sat_ratio


def compute_sat_ratio_bound(count_unsat_con_sm_eq, count_unsat_con_sm_st, count_unsat_con_la_eq, count_unsat_con_la_st, count_total):
    """Compute satisfaction ratio of the bound constraints.

    The satisfaction ratio is defined as the number of constraints are satisfied divided by the total number of constraints. The constraints
    are counted for each training example individually. For example if there are 5 bound constraints and one has 2 training examples, then
    the total number of constraints is 5*2=10 and the number of constraints satisfied is the sum of constraints that are satisfied by the
    first training example and the second example.

    Parameters
    ----------
    count_unsat_con_sm_eq : tf.Tensor
        number of unsatisfied smaller or equal constraints
    count_unsat_con_sm_st : tf.Tensor
        number of unsatisfied strictly smaller constraints
    count_unsat_con_la_eq : tf.Tensor
        number of unsatisfied larger or equal constraints
    count_unsat_con_la_st : tf.Tensor
        number of unsatisfied strictly larger constraints
    count_total : tf.Tensor
        total number of constraints for the current batch

    Returns
    ----------
    sat_ratio : tf.Tensor
        Float indicating the percentage of constraints satisfied for the current batch.
    """
    total_unsat_con = tf.math.add_n([count_unsat_con_sm_eq, count_unsat_con_sm_st, count_unsat_con_la_eq, count_unsat_con_la_st])

    sat_ratio = tf.math.subtract(count_total, total_unsat_con)
    sat_ratio = tf.math.divide(sat_ratio, count_total)

    return sat_ratio


def max_sat_ratio(dataset_output, bound_constraints, signed_indicator):
    """Compute the satisfaction ratio of the whole training set.

    This function uses the functions construct_bound_con, determine_sat_con and compute_sat_ratio.

    The function computes the satisfaction ratio using the groundtruth of the training examples. This allows for detecting noise in the
    dataset that is not in line with the given constraints.

    Parameters
    ----------
    dataset_output : numpy.ndarray
        The output of all the training examples.
    bound_constraints : list[list[tf.Tensor]]
        list of the bound constraints, where each bound constraint consists of 4 tensors: output index, value of bound, smaller or larger
        string, and equal or strict string.
    signed_indicator : tf.Tensor
        Tensor of shape=(batch_size, output_dim) and dtype=tf.float32 containing the signs of the output variables in the sum constraint.

    Returns
    ----------
    max_sat_rat : tf.Tensor
        Satisfaction ratio of the whole training set.

    """
    data_size = np.size(dataset_output, 0)
    output_dim = np.size(dataset_output, 1)
    bound_sm_eq, bound_sm_st, bound_la_eq, bound_la_st, ind_sm_eq, ind_sm_st, ind_la_eq, ind_la_st = \
        construct_bound_con(bound_constraints, data_size, output_dim)

    dataset_output = tf.constant(dataset_output, dtype=tf.float32)
    _, max_count_unsat_sm_eq, max_count_sm_eq = determine_sat_con(outputs=dataset_output,
                                                                  bound_constraints=bound_sm_eq,
                                                                  batch_size=data_size,
                                                                  output_dim=output_dim,
                                                                  type_constraint=tf.constant('se', dtype=tf.string),
                                                                  ind_sm_eq=ind_sm_eq,
                                                                  ind_sm_st=ind_sm_st,
                                                                  ind_la_eq=ind_la_eq,
                                                                  ind_la_st=ind_la_st)
    _, max_count_unsat_sm_st, max_count_sm_st = determine_sat_con(outputs=dataset_output,
                                                                  bound_constraints=bound_sm_st,
                                                                  batch_size=data_size,
                                                                  output_dim=output_dim,
                                                                  type_constraint=tf.constant('ss', dtype=tf.string),
                                                                  ind_sm_eq=ind_sm_eq,
                                                                  ind_sm_st=ind_sm_st,
                                                                  ind_la_eq=ind_la_eq,
                                                                  ind_la_st=ind_la_st)
    _, max_count_unsat_la_eq, max_count_la_eq = determine_sat_con(outputs=dataset_output,
                                                                  bound_constraints=bound_la_eq,
                                                                  batch_size=data_size,
                                                                  output_dim=output_dim,
                                                                  type_constraint=tf.constant('le', dtype=tf.string),
                                                                  ind_sm_eq=ind_sm_eq,
                                                                  ind_sm_st=ind_sm_st,
                                                                  ind_la_eq=ind_la_eq,
                                                                  ind_la_st=ind_la_st)
    _, max_count_unsat_la_st, max_count_la_st = determine_sat_con(outputs=dataset_output,
                                                                  bound_constraints=bound_la_st,
                                                                  batch_size=data_size,
                                                                  output_dim=output_dim,
                                                                  type_constraint=tf.constant('sl', dtype=tf.string),
                                                                  ind_sm_eq=ind_sm_eq,
                                                                  ind_sm_st=ind_sm_st,
                                                                  ind_la_eq=ind_la_eq,
                                                                  ind_la_st=ind_la_st)

    # added the satisfaction check for the sum constraint
    sum_expression = sum_expression_output_variables(signed_indicator, dataset_output)

    _, max_satisfaction_signed_indicator, max_counter_unsatisfaction_sum, max_counter_total_sum = \
        check_sum_expression(sum_expression,
                             tf.constant('se', dtype=tf.string),
                             tf.constant(0, dtype=tf.float32, shape=(data_size, 1)),
                             signed_indicator,
                             data_size)

    count_con_total = [max_count_sm_eq, max_count_sm_st, max_count_la_eq, max_count_la_st, max_counter_total_sum]
    count_con_unsat = [max_count_unsat_sm_eq, max_count_unsat_sm_st, max_count_unsat_la_eq, max_count_unsat_la_st,
                       max_counter_unsatisfaction_sum]

    max_sat_rat = compute_sat_ratio(list_counter_unsatisfied=count_con_unsat, list_counter_total=count_con_total)

    return max_sat_rat


def max_sat_ratio_bound(dataset_output, bound_constraints, output_dim):
    """Compute the satisfaction ratio of the bound constraints on the whole training set.

    This function uses the functions construct_bound_con, determine_sat_con and compute_sat_ratio.

    Parameters
    ----------
    dataset_output : numpy.ndarray
        The output of all the training examples.
    bound_constraints : list[list[tf.Tensor]]
        list of the bound constraints, where each bound constraint consists of 4 tensors: output index, value of bound, smaller or larger
        string, and equal or strict string.
    output_dim : int
        output dimension of the network

    Returns
    ----------
    max_sat_rat : tf.Tensor
        Satisfaction ratio of the whole training set.

    """
    data_size = np.size(dataset_output, 0)
    bound_sm_eq, bound_sm_st, bound_la_eq, bound_la_st, ind_sm_eq, ind_sm_st, ind_la_eq, ind_la_st = \
        construct_bound_con(bound_constraints, data_size, output_dim)
    dataset_output = tf.constant(dataset_output, dtype=tf.float32)
    _, max_count_unsat_sm_eq, max_count_sm_eq = determine_sat_con(outputs=dataset_output,
                                                                  bound_constraints=bound_sm_eq,
                                                                  batch_size=data_size,
                                                                  output_dim=output_dim,
                                                                  type_constraint=tf.constant('se', dtype=tf.string),
                                                                  ind_sm_eq=ind_sm_eq,
                                                                  ind_sm_st=ind_sm_st,
                                                                  ind_la_eq=ind_la_eq,
                                                                  ind_la_st=ind_la_st)
    _, max_count_unsat_sm_st, max_count_sm_st = determine_sat_con(outputs=dataset_output,
                                                                  bound_constraints=bound_sm_st,
                                                                  batch_size=data_size,
                                                                  output_dim=output_dim,
                                                                  type_constraint=tf.constant('ss', dtype=tf.string),
                                                                  ind_sm_eq=ind_sm_eq,
                                                                  ind_sm_st=ind_sm_st,
                                                                  ind_la_eq=ind_la_eq,
                                                                  ind_la_st=ind_la_st)
    _, max_count_unsat_la_eq, max_count_la_eq = determine_sat_con(outputs=dataset_output,
                                                                  bound_constraints=bound_la_eq,
                                                                  batch_size=data_size,
                                                                  output_dim=output_dim,
                                                                  type_constraint=tf.constant('le', dtype=tf.string),
                                                                  ind_sm_eq=ind_sm_eq,
                                                                  ind_sm_st=ind_sm_st,
                                                                  ind_la_eq=ind_la_eq,
                                                                  ind_la_st=ind_la_st)
    _, max_count_unsat_la_st, max_count_la_st = determine_sat_con(outputs=dataset_output,
                                                                  bound_constraints=bound_la_st,
                                                                  batch_size=data_size,
                                                                  output_dim=output_dim,
                                                                  type_constraint=tf.constant('sl', dtype=tf.string),
                                                                  ind_sm_eq=ind_sm_eq,
                                                                  ind_sm_st=ind_sm_st,
                                                                  ind_la_eq=ind_la_eq,
                                                                  ind_la_st=ind_la_st)

    max_count_total = tf.math.add_n([max_count_sm_eq, max_count_sm_st, max_count_la_eq, max_count_la_st])

    max_sat_rat = compute_sat_ratio_bound(max_count_unsat_sm_eq,
                                          max_count_unsat_sm_st,
                                          max_count_unsat_la_eq,
                                          max_count_unsat_la_st,
                                          max_count_total)

    return max_sat_rat


def sum_output_variables(index_positive_output, index_negative_output, batch_size, output_dim):
    """Construct the sum of output variables that together need to satisfy a bound constraint.

    This function constructs an indicator tensor that can be used to multiply element-wise with the predictions of the model. In order to
    construct the bound constraint of this expression, one should add the result of this element-wise multiplication along the rows (so for
    each training examples the result should be added) and this sum should be checked for being smaller/larger and equal/strict.

    The convention for positive and negative signs is as follows. The variables that are on the left-hand side of the inequality constraint
    have a positive sign and the others have a negative sign. For example if one has the constraint $y_1+y_2 \leq y_5 + 4$, then the
    variables $y_1,y_2$ have a positive sign, the variable $y_5$ has a negative sign, the constraint is smaller or equal and the constant
    is 4. Of course this constraint can also be interpreted as $y_5$ being the positive variable, the variables $y_1,y_2$ as the negative
    variables, the constraint is larger or equal and the constant is -4. This is completely equivalent, but the most important thing to
    note is that once one has fixed a left-hand side and right-hand side than this translation is well-defined. Indeed, in our example we
    have switched the left-hand side and the right-hand side to find the other equivalent statement. This allows for a fixed interpretation
    of the bound constraints in this case.

    Parameters
    ----------
    index_positive_output : list[tf.Tensor]
        A list of tensors indicating which output variables have a positive sign in the sum. The indices are assumed to be of
        dtype=tf.int32.
    index_negative_output: list[tf.Tensor]
        A list of tensors indicating which output variables have a negative sign in the sum. The indices are assumed to be of
        dtype=tf.int32.
    batch_size : int
        An integer tensor indicating the batch size used during training.
    output_dim : int
        An integer tensor indicating the output dimension of the network.

    Returns
    ----------
    signed_indicator : tf.Tensor
        A constant tensor consisting of 0, -1 and 1 indicating the sign of the corresponding output variable in the bound constraint. This
        tensor has dtype=tf.float32 and shape=(batch_size, output_dim).

    """
    signed_indicator = tf.zeros(shape=[batch_size, output_dim], dtype=tf.float32)

    for i in range(0, len(index_positive_output)):
        bound_constraint, indicator_constraint = convert_bound_con(index_positive_output[i],
                                                                   tf.constant(1, dtype=tf.float32),
                                                                   batch_size=batch_size,
                                                                   output_dim=output_dim)
        signed_indicator = tf.math.add(signed_indicator, bound_constraint)

    for i in range(0, len(index_negative_output)):
        bound_constraint, indicator_constraint = convert_bound_con(index_negative_output[i],
                                                                   tf.constant(-1, dtype=tf.float32),
                                                                   batch_size=batch_size,
                                                                   output_dim=output_dim)
        signed_indicator = tf.math.add(signed_indicator, bound_constraint)

    return signed_indicator


def sum_expression_output_variables(signed_indicator, predictions, upper_bound_norm_output_variables, lower_bound_norm_output_variables):
    """This function preforms the sum over the relevant output variables.

    In other words, this function can be used to in the GradientTape such that the sum expression can be automatically differentiated.

    Parameters
    ----------
    signed_indicator : tf.Tensor
        constant tensor of dtype=tf.float32 and shape=(batch_size, output_dimension) that consists of 1, 0 and -1 (all with
        dtype=tf.float32) indicating the sign of the corresponding output variable in the sum.
    predictions: tf.Tensor
        constant tensor of dtype=tf.float32 and shape=(batch_size, output_dimension) which denotes the output of the network.
    upper_bound_norm_output_variables : tf.Tensor
        Tensor of shape=(1, output_dim) and dtype=tf.float32 containing the upper bound used for the normalization of the output variables.
    lower_bound_norm_output_variables : tf.Tensor
        Tensor of shape=(1, output_dim) and dtype=tf.float32 containing the lower bound used for the normalization of the output variables.

    Returns
    ----------
    sum_bound_constraint : tf.Tensor
        constant tensor of dtype=tf.float32 and shape=(batch_size,) which contains in each element the sum of all the predictions with the
        correct sign for each training example separately.
    """
    signed_indicated_predictions = tf.math.multiply(signed_indicator, predictions)
    rescaled_signed_indicated_predictions = tf.math.add(tf.math.multiply(tf.math.subtract(upper_bound_norm_output_variables,
                                                                                          lower_bound_norm_output_variables),
                                                                         signed_indicated_predictions),
                                                        lower_bound_norm_output_variables)

    sum_bound_constraint = tf.math.reduce_sum(rescaled_signed_indicated_predictions, axis=1)

    return sum_bound_constraint


def check_sum_expression(expression, type_constraint, bound, signed_indicator, batch_size, undo_norm_direction):
    """Determine indicator tensor that indicates which training examples satisfy the sum constraint (0) and which do not satisfy (1). At the
    same time, adjust the indicator function of the expressions to adjust for this satisfaction.

    This function checks whether or not a training example satisfies the sum constraint defined by expression and bound.

    Parameters
    ----------
    expression : tf.Tensor
        constant tensor of dtype=tf.float32 and shape=(batch_size,) which contains a linear sum of the output variables with a potential
        change of sign of some variables
    type_constraint: tf.Tensor
        constant tensor of dtype=tf.string which denotes the type of inequality constraint that is needed for expression. This input should
        be always a tensor of the corresponding dtype and equal to 'se', 'ss', 'le' or 'ls'. The abbreviations stand for smaller or equal,
        strictly smaller, larger or equal and strictly larger, respectively.
    bound : tf.Tensor
        constant tensor of dtype=tf.float32 and shape=(batch_size,) which contains the constant part of the constraint corresponding to
        expression.
    signed_indicator : tf.Tensor
        constant tensor of dtype=tf.float32 and shape=(batch_size,output_dimension) containing the signs of the output variables as they
        appear in expression
    batch_size : int
        integer denoting the batch_size used in training
    undo_norm_direction : tf.Tensor
        constant tensor of dtype=tf.float32 and shape(1, output_dimension) containing the factor used to do the normalization of the output
        variables.

    Returns
    ----------
    unsatisfied_constraint : tf.Tensor
        constant tensor of dtype=tf.float32 and shape=(batch_size, 1) consisting of 1s and 0s. A value 1 indicates that the constraint is
        not satisfied for this training example and the value 0 indicates that this training example satisfies this constraint.
    satisfaction_signed_indicator : tf.Tensor
        constant tensor of dtype=tf.float32 and shape=(batch_size, output_dimension) that is equal to signed_indicator except that for the
        rows of the training examples that satisfy the constraint. These rows are replaced by 0s.
    counter_unsatisfaction : tf.Tensor
        constant tensor of dtype=tf.float32 and shape=(,) denoting how many training examples do not satisfy the constraint.
    counter_total_sum_constraint : tf.Tensor
        constant tensor of dtype=tf.float32 and shape=(,) denoting how many training examples are used and thus the maximal number of
        training examples that could satisfy the constraint.

    """
    bound = tf.reshape(bound, [batch_size, 1])  # make sure that the bound is of the correct shape
    expression = tf.reshape(expression, [batch_size, 1])
    if type_constraint == tf.constant('se', dtype=tf.string):
        satisfaction_constraint = tf.math.less_equal(expression, bound)
    elif type_constraint == tf.constant('ss', dtype=tf.string):
        satisfaction_constraint = tf.math.less(expression, bound)
    elif type_constraint == tf.constant('le', dtype=tf.string):
        satisfaction_constraint = tf.math.greater(expression, bound)
    elif type_constraint == tf.constant('ls', dtype=tf.string):
        satisfaction_constraint = tf.math.greater_equal(expression, bound)
    else:
        satisfaction_constraint = tf.constant(True, dtype=tf.bool, shape=[batch_size, 1])
        tf.print('Invalid type of sum constraint.')

    unsatisfied_constraint = tf.where(satisfaction_constraint, tf.zeros(shape=[batch_size, 1], dtype=tf.float32), tf.ones(shape=[batch_size, 1], dtype=tf.float32))
    unsatisfied_constraint = tf.reshape(unsatisfied_constraint, [batch_size, 1])

    satisfaction_signed_indicator = tf.math.multiply(tf.math.multiply(unsatisfied_constraint, signed_indicator), undo_norm_direction)

    counter_total_sum_constraint = tf.constant(batch_size, dtype=tf.float32)

    counter_unsatisfaction = tf.math.reduce_sum(unsatisfied_constraint, axis=None)

    return unsatisfied_constraint, satisfaction_signed_indicator, counter_unsatisfaction, counter_total_sum_constraint


def perform_satisfaction_expression(satisfaction_signed_indicator, predictions):
    """Perform the sum that can be recorded by the GradientTape.

    Parameters
    ----------
    satisfaction_signed_indicator : tf.Tensor
        constant tensor containing the sign of each output variable in the sum. By convention, the sign of a variable that does not occur
        in the sum is put to 0.
    predictions : tf.Tensor
        tensor containing the output of the model for the current batch.

    Returns
    ----------
    unsatisfied_expression : tf.Tensor
        constant tensor containing the sum expressions for only those training examples that do not satisfy the sum constraint.

    """
    unsatisfied_predictions = tf.math.multiply(satisfaction_signed_indicator, predictions)

    unsatisfied_expression = tf.math.reduce_sum(unsatisfied_predictions, axis=1, keepdims=True)

    return unsatisfied_expression


def compute_sum_grad_undo_norm(model, inputs, targets, batch_size, output_dim, neur_act, bound_sm_eq, bound_sm_st, bound_la_eq, bound_la_st,
                               ind_sm_eq, ind_sm_st, ind_la_eq, ind_la_st, signed_indicator_food_expense, signed_indicator_income,
                               lower_bound_output_variables, upper_bound_output_variables, lower_bound_input_variables,
                               upper_bound_input_variables, undo_norm_direction):
    """Compute the gradients of the unsatisfied constraints and the loss function.

    This function uses determine_sat_con, compute_sat_ratio, check_sum_expression and perform_satisfaction_expression. For more info,
    please read the documentation of these functions.

    The gradient of the loss function and each type of constraints individually is computed. Note that the satisfaction of the constraints
    is checked inside the stop_recording of the GradientTape. This is necessary because there are no differentiable conditionals. Therefore
    the indicator tensors are used to remove the satisfied constraints as well as the variables that are unconstrained. If one wants to add
    additional kind of constraints, then one should check their satisfaction in a similar way and one should make use of indicator tensors
    and @tf.function s (not depending on conditionals) to construct them in the recording of the GradientTape.

    In addition, the satisfaction ratio is returned as well as the value of the loss function and metrics.

    Note that for expression1 and expression2, some parameters are re-used. The reason for this is that the factor under the square root in
    both formulas is exactly the same. If one wants to use constraints that are completely different, then one should include for each
    formula the indicator, ignore and factor. Moreover, if a given formula consists of multiple fractions then it could occur that multiple
    of these parameters need to be used in order to define the constraint correctly.

    Parameters
    ----------
    model : tf.keras.Model
        model that is trained
    inputs : tf.Tensor
        x part of tf.data.Dataset
    targets : tf.Tensor
        y part of tf.data.Dataset
    batch_size : int
        batch-size used during training
    output_dim : int
        output dimension of the network
    neur_act : list[int]
        list of integers denoting the number of neurons or activation functions in the layers of the network.
    bound_sm_eq : tf.Tensor
        consists of all the columns with bounds corresponding to smaller or equal constraints on the output. This tensor has
        dtype=tf.float32 and shape=(batch_size, output_dim).
    bound_sm_st : tf.Tensor
        similarly defined as previous but for the strictly smaller constraints. This tensor has dtype=tf.float32 and
        shape=(batch_size, output_dim).
    bound_la_eq : tf.Tensor
        similarly defined as previous but for the larger or equal constraints. This tensor has dtype=tf.float32 and
        shape=(batch_size, output_dim).
    bound_la_st : tf.Tensor
        similarly defined as previous but for the strictly larger constraints. This tensor has dtype=tf.float32 and
        shape=(batch_size, output_dim).
    ind_sm_eq : tf.Tensor
        corresponding indicator tensor of bound_sm_eq. This tensor has dtype=tf.float32 and shape=(batch_size, output_dim).
    ind_sm_st : tf.Tensor
        similarly defined as ind_sm_eq but for the strictly smaller constraints. This tensor has dtype=tf.float32 and
        shape=(batch_size, output_dim).
    ind_la_eq : tf.Tensor
        similarly defined as ind_sm_eq but for the larger or equal constraints. This tensor has dtype=tf.float32 and
        shape=(batch_size, output_dim).
    ind_la_st : tf.Tensor
        similarly defined as ind_sm_eq but for the strictly larger constraints. This tensor has dtype=tf.float32 and
        shape=(batch_size, output_dim).
    signed_indicator_food_expense : tf.Tensor
        constant tensor consisting of -1, 0 and 1 indicating the presence of an output variable in the constraint and the sign if this
        variable is present. It is assumed that all variables are written on the left-hand side of the smaller or equal sign. If the sum
        constraint would have been strict, then it is assumed that all the variables are written on the left-hand side of the strictly
        smaller sign but then the code used for checking the constraint should also be slightly adjusted.
    signed_indicator_income : tf.Tensor
        constant tensor consisting of -1, 0 and 1 indicating the presence of an output variable in the constraint and the sign if this
        variable is present. It is assumed that all variables are written on the left-hand side of the smaller or equal sign. If the sum
        constraint would have been strict, then it is assumed that all the variables are written on the left-hand side of the strictly
        smaller sign but then the code used for checking the constraint should also be slightly adjusted.
    lower_bound_output_variables : tf.Tensor
        Tensor of shape=(1, output_dimension) and dtype=tf.float32 denoting the lower bound used for the normalization of each output
        variable.
    upper_bound_output_variables : tf.Tensor
        Tensor of shape=(1, output_dimension) and dtype=tf.float32 denoting the upper bound used for the normalization of each output
        variable.
    lower_bound_input_variables : tf.Tensor
        Tensor of shape=(1, input_dimension) and dtype=tf.float32 denoting the lower bound used for the normalization of each input
        variable.
    upper_bound_input_variables : tf.Tensor
        Tensor of shape=(1, input_dimension) and dtype=tf.float32 denoting the upper bound used for the normalization of each input
        variable.
    undo_norm_direction : tf.Tensor
        constant tensor of dtype=tf.float32 and shape(1, output_dimension) containing the factor used to do the normalization of the output
        variables.

    Returns
    ----------
    grad_loss : list[tf.Tensor]
        list of tensors corresponding to the gradient of the loss with respect to the trainable variables of model
    grad_sm_eq : list[tf.Tensor]
        list of tensors corresponding to the gradient of the unsatisfied smaller or equal constraints with respect to the trainable
        variables of model
    grad_sm_st : list[tf.Tensor]
        list of tensors corresponding to the gradient of the unsatisfied strictly smaller constraints with respect to the trainable
        variables of model
    grad_la_eq : list[tf.Tensor]
        list of tensors corresponding to the gradient of the unsatisfied larger or equal constraints with respect to the trainable
        variables of model
    grad_la_st : list[tf.Tensor]
        list of tensors corresponding to the gradient of the unsatisfied strictly larger constraints with respect to the trainable
        variables of model
    grad_sum_food_expense : list[tf.Tensor]
        list of tensors corresponding to the gradient of the unsatisfied sum constraint food expense with respect to the trainable
        variables of model
    grad_sum_income : list[tf.Tensor]
        list of tensors corresponding to the gradient of the unsatisfied sum constraint income with respect to the trainable variables of
        model
    loss_value : tf.Tensor
        value of loss function of the current batch
    metric_value : tf.Tensor
        value of the metric of the current batch
    sat_ratio : tf.Tensor
        satisfaction ratio of the constraints for the current batch

    """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        tape_pred = model(inputs, training=True)  # forward pass to have it recorded by compute_tape
        loss_value = tf.keras.losses.mean_squared_error(targets, tape_pred)

        with tape.stop_recording():
            metric_value = tf.keras.losses.mean_absolute_error(targets,
                                                               tape_pred)  # the metric does not need to be differentiated and thus not necessary to record
            # check satisfaction for smaller or equal constraints
            ind_sat_sm_eq, count_unsat_sm_eq, count_sm_eq = determine_sat_con(tape_pred,
                                                                              bound_sm_eq,
                                                                              batch_size,
                                                                              output_dim,
                                                                              tf.constant('se', dtype=tf.string),
                                                                              ind_sm_eq,
                                                                              ind_sm_st,
                                                                              ind_la_eq,
                                                                              ind_la_st)
            # check satisfaction for strictly smaller constraints
            ind_sat_sm_st, count_unsat_sm_st, count_sm_st = determine_sat_con(tape_pred,
                                                                              bound_sm_st,
                                                                              batch_size,
                                                                              output_dim,
                                                                              tf.constant('ss', dtype=tf.string),
                                                                              ind_sm_eq,
                                                                              ind_sm_st,
                                                                              ind_la_eq,
                                                                              ind_la_st)
            # check satisfaction for larger or equal constraints
            ind_sat_la_eq, count_unsat_la_eq, count_la_eq = determine_sat_con(tape_pred,
                                                                              bound_la_eq,
                                                                              batch_size,
                                                                              output_dim,
                                                                              tf.constant('le', dtype=tf.string),
                                                                              ind_sm_eq,
                                                                              ind_sm_st,
                                                                              ind_la_eq,
                                                                              ind_la_st)
            # check satisfaction for strictly larger constraints
            ind_sat_la_st, count_unsat_la_st, count_la_st = determine_sat_con(tape_pred,
                                                                              bound_la_st,
                                                                              batch_size,
                                                                              output_dim,
                                                                              tf.constant('sl', dtype=tf.string),
                                                                              ind_sm_eq,
                                                                              ind_sm_st,
                                                                              ind_la_eq,
                                                                              ind_la_st)

            #  satisfaction check for the sum constraint food expense
            sum_expression_food_expense = sum_expression_output_variables(signed_indicator_food_expense,
                                                                          tape_pred,
                                                                          upper_bound_output_variables,
                                                                          lower_bound_output_variables)

            unsatisfied_sum_constraint_food_expense, satisfaction_signed_indicator_food_expense, counter_unsatisfaction_sum_food_expense, counter_total_sum_food_expense = \
                check_sum_expression(sum_expression_food_expense,
                                     tf.constant('se', dtype=tf.string),
                                     tf.constant(0, dtype=tf.float32, shape=(batch_size, 1)),
                                     signed_indicator_food_expense,
                                     batch_size,
                                     undo_norm_direction)

            # satisfaction check for the sum constraint income
            sum_expression_income = sum_expression_output_variables(signed_indicator_income,
                                                                    tape_pred,
                                                                    upper_bound_output_variables,
                                                                    lower_bound_output_variables)

            unsatisfied_sum_constraint_income, satisfaction_signed_indicator_income, counter_unsatisfaction_sum_income, counter_total_sum_income = \
                check_sum_expression(sum_expression_income,
                                     tf.constant('se', dtype=tf.string),
                                     # extract first column of inputs tensor and undo the normalization
                                     #tf.math.add(tf.math.multiply(tf.slice(input_=inputs, begin=[0, 0], size=[tf.shape(inputs)[0], 1]),
                                     tf.math.add(tf.math.multiply(tf.reshape(tf.slice(input_=inputs, begin=[0, 0], size=[-1, 1]), [batch_size, 1]),
                                                                  tf.math.subtract(tf.gather_nd(upper_bound_input_variables, (0, 0)),
                                                                                   tf.gather_nd(lower_bound_input_variables, (0, 0)))),
                                                 tf.gather_nd(lower_bound_input_variables, (0, 0))),
                                     signed_indicator_income,
                                     batch_size,
                                     undo_norm_direction)


            # put total number of constraints for each type in a list
            count_con_total = [count_sm_eq, count_sm_st, count_la_eq, count_la_st, counter_total_sum_food_expense, counter_total_sum_income]
            # put total number of unsatisfied constraints for each type in a list
            count_con_unsat = [count_unsat_sm_eq, count_unsat_sm_st, count_unsat_la_eq, count_unsat_la_st,
                               counter_unsatisfaction_sum_food_expense, counter_unsatisfaction_sum_income]

        # put output variables that are not constrained by the specific type to zero as well as the training examples that satisfy the constraint
        unsat_con_sm_eq = tf.math.multiply(tape_pred, ind_sat_sm_eq)
        unsat_con_sm_st = tf.math.multiply(tape_pred, ind_sat_sm_st)
        unsat_con_la_eq = tf.math.multiply(tape_pred, ind_sat_la_eq)
        unsat_con_la_st = tf.math.multiply(tape_pred, ind_sat_la_st)

        # for the unsatisfied sum constraints, perform the sum
        unsatisfied_sum_food_expense = perform_satisfaction_expression(satisfaction_signed_indicator_food_expense, tape_pred)
        unsatisfied_sum_income = perform_satisfaction_expression(satisfaction_signed_indicator_income, tape_pred)

    # compute gradient for loss function
    grad_loss = tape.gradient(loss_value, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    # compute gradient for the smaller or equal constraints
    grad_sm_eq = tape.gradient(unsat_con_sm_eq, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    # compute gradient for the strictly smaller constraints
    grad_sm_st = tape.gradient(unsat_con_sm_st, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    # compute gradient for the larger or equal constraints
    grad_la_eq = tape.gradient(unsat_con_la_eq, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    # compute gradient for the stirctly larger constraints
    grad_la_st = tape.gradient(unsat_con_la_st, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    # compute gradient for the sum constraint food expense
    grad_sum_food_expense = tape.gradient(unsatisfied_sum_food_expense,
                                          model.trainable_variables,
                                          unconnected_gradients=tf.UnconnectedGradients.ZERO)
    # compute gradient for the sum constraint income
    grad_sum_income = tape.gradient(unsatisfied_sum_income, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)

    del tape  # delete the gradient tape

    sat_ratio = compute_sat_ratio(count_con_unsat, count_con_total)  # compute the satisfaction ratio for the current model (before
    # applying gradients)

    return grad_loss, grad_sm_eq, grad_sm_st, grad_la_eq, grad_la_st, grad_sum_food_expense, grad_sum_income, loss_value, metric_value, sat_ratio


def unsup_compute_sum_grad_undo_norm(model, inputs, targets, batch_size, output_dim, neur_act, bound_sm_eq, bound_sm_st, bound_la_eq,
                                     bound_la_st, ind_sm_eq, ind_sm_st, ind_la_eq, ind_la_st, signed_indicator_food_expense,
                                     signed_indicator_income, lower_bound_output_variables, upper_bound_output_variables,
                                     lower_bound_input_variables, upper_bound_input_variables, undo_norm_direction):
    """Compute the gradients of the unsatisfied constraints and the loss function.

    This function uses determine_sat_con, compute_sat_ratio, check_sum_expression and perform_satisfaction_expression. For more info,
    please read the documentation of these functions.

    The gradient of the loss function and each type of constraints individually is computed. Note that the satisfaction of the constraints
    is checked inside the stop_recording of the GradientTape. This is necessary because there are no differentiable conditionals. Therefore
    the indicator tensors are used to remove the satisfied constraints as well as the variables that are unconstrained. If one wants to add
    additional kind of constraints, then one should check their satisfaction in a similar way and one should make use of indicator tensors
    and @tf.function s (not depending on conditionals) to construct them in the recording of the GradientTape.

    In addition, the satisfaction ratio is returned as well as the value of the loss function and metrics.

    Note that for expression1 and expression2, some parameters are re-used. The reason for this is that the factor under the square root in
    both formulas is exactly the same. If one wants to use constraints that are completely different, then one should include for each
    formula the indicator, ignore and factor. Moreover, if a given formula consists of multiple fractions then it could occur that multiple
    of these parameters need to be used in order to define the constraint correctly.

    Parameters
    ----------
    model : tf.keras.Model
        model that is trained
    inputs : tf.Tensor
        x part of tf.data.Dataset
    targets : tf.Tensor
        y part of tf.data.Dataset
    batch_size : int
        batch-size used during training
    output_dim : int
        output dimension of the network
    neur_act : list[int]
        list of integers denoting the number of neurons or activation functions in the layers of the network.
    bound_sm_eq : tf.Tensor
        consists of all the columns with bounds corresponding to smaller or equal constraints on the output. This tensor has
        dtype=tf.float32 and shape=(batch_size, output_dim).
    bound_sm_st : tf.Tensor
        similarly defined as previous but for the strictly smaller constraints. This tensor has dtype=tf.float32 and
        shape=(batch_size, output_dim).
    bound_la_eq : tf.Tensor
        similarly defined as previous but for the larger or equal constraints. This tensor has dtype=tf.float32 and
        shape=(batch_size, output_dim).
    bound_la_st : tf.Tensor
        similarly defined as previous but for the strictly larger constraints. This tensor has dtype=tf.float32 and
        shape=(batch_size, output_dim).
    ind_sm_eq : tf.Tensor
        corresponding indicator tensor of bound_sm_eq. This tensor has dtype=tf.float32 and shape=(batch_size, output_dim).
    ind_sm_st : tf.Tensor
        similarly defined as ind_sm_eq but for the strictly smaller constraints. This tensor has dtype=tf.float32 and
        shape=(batch_size, output_dim).
    ind_la_eq : tf.Tensor
        similarly defined as ind_sm_eq but for the larger or equal constraints. This tensor has dtype=tf.float32 and
        shape=(batch_size, output_dim).
    ind_la_st : tf.Tensor
        similarly defined as ind_sm_eq but for the strictly larger constraints. This tensor has dtype=tf.float32 and
        shape=(batch_size, output_dim).
    signed_indicator_food_expense : tf.Tensor
        constant tensor consisting of -1, 0 and 1 indicating the presence of an output variable in the constraint and the sign if this
        variable is present. It is assumed that all variables are written on the left-hand side of the smaller or equal sign. If the sum
        constraint would have been strict, then it is assumed that all the variables are written on the left-hand side of the strictly
        smaller sign but then the code used for checking the constraint should also be slightly adjusted.
    signed_indicator_income : tf.Tensor
        constant tensor consisting of -1, 0 and 1 indicating the presence of an output variable in the constraint and the sign if this
        variable is present. It is assumed that all variables are written on the left-hand side of the smaller or equal sign. If the sum
        constraint would have been strict, then it is assumed that all the variables are written on the left-hand side of the strictly
        smaller sign but then the code used for checking the constraint should also be slightly adjusted.
    lower_bound_output_variables : tf.Tensor
        Tensor of shape=(1, output_dimension) and dtype=tf.float32 denoting the lower bound used for the normalization of each output
        variable.
    upper_bound_output_variables : tf.Tensor
        Tensor of shape=(1, output_dimension) and dtype=tf.float32 denoting the upper bound used for the normalization of each output
        variable.
    lower_bound_input_variables : tf.Tensor
        Tensor of shape=(1, input_dimension) and dtype=tf.float32 denoting the lower bound used for the normalization of each input
        variable.
    upper_bound_input_variables : tf.Tensor
        Tensor of shape=(1, input_dimension) and dtype=tf.float32 denoting the upper bound used for the normalization of each input
        variable.
    undo_norm_direction : tf.Tensor
        constant tensor of dtype=tf.float32 and shape(1, output_dimension) containing the factor used to do the normalization of the output
        variables.

    Returns
    ----------
    grad_loss : list[tf.Tensor]
        list of tensors corresponding to the gradient of the loss with respect to the trainable variables of model
    grad_sm_eq : list[tf.Tensor]
        list of tensors corresponding to the gradient of the unsatisfied smaller or equal constraints with respect to the trainable
        variables of model
    grad_sm_st : list[tf.Tensor]
        list of tensors corresponding to the gradient of the unsatisfied strictly smaller constraints with respect to the trainable
        variables of model
    grad_la_eq : list[tf.Tensor]
        list of tensors corresponding to the gradient of the unsatisfied larger or equal constraints with respect to the trainable
        variables of model
    grad_la_st : list[tf.Tensor]
        list of tensors corresponding to the gradient of the unsatisfied strictly larger constraints with respect to the trainable
        variables of model
    grad_sum_food_expense : list[tf.Tensor]
        list of tensors corresponding to the gradient of the unsatisfied sum constraint food expense with respect to the trainable
        variables of model
    grad_sum_income : list[tf.Tensor]
        list of tensors corresponding to the gradient of the unsatisfied sum constraint income with respect to the trainable variables of
        model
    sat_ratio : tf.Tensor
        satisfaction ratio of the constraints for the current batch

    """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        tape_pred = model(inputs, training=True)  # forward pass to have it recorded by compute_tape

        with tape.stop_recording():
            metric_value = tf.keras.losses.mean_absolute_error(targets,
                                                               tape_pred)  # the metric does not need to be differentiated and thus not necessary to record
            # check satisfaction for smaller or equal constraints
            ind_sat_sm_eq, count_unsat_sm_eq, count_sm_eq = determine_sat_con(tape_pred,
                                                                              bound_sm_eq,
                                                                              batch_size,
                                                                              output_dim,
                                                                              tf.constant('se', dtype=tf.string),
                                                                              ind_sm_eq,
                                                                              ind_sm_st,
                                                                              ind_la_eq,
                                                                              ind_la_st)
            # check satisfaction for strictly smaller constraints
            ind_sat_sm_st, count_unsat_sm_st, count_sm_st = determine_sat_con(tape_pred,
                                                                              bound_sm_st,
                                                                              batch_size,
                                                                              output_dim,
                                                                              tf.constant('ss', dtype=tf.string),
                                                                              ind_sm_eq,
                                                                              ind_sm_st,
                                                                              ind_la_eq,
                                                                              ind_la_st)
            # check satisfaction for larger or equal constraints
            ind_sat_la_eq, count_unsat_la_eq, count_la_eq = determine_sat_con(tape_pred,
                                                                              bound_la_eq,
                                                                              batch_size,
                                                                              output_dim,
                                                                              tf.constant('le', dtype=tf.string),
                                                                              ind_sm_eq,
                                                                              ind_sm_st,
                                                                              ind_la_eq,
                                                                              ind_la_st)
            # check satisfaction for strictly larger constraints
            ind_sat_la_st, count_unsat_la_st, count_la_st = determine_sat_con(tape_pred,
                                                                              bound_la_st,
                                                                              batch_size,
                                                                              output_dim,
                                                                              tf.constant('sl', dtype=tf.string),
                                                                              ind_sm_eq,
                                                                              ind_sm_st,
                                                                              ind_la_eq,
                                                                              ind_la_st)

            #  satisfaction check for the sum constraint food expense
            sum_expression_food_expense = sum_expression_output_variables(signed_indicator_food_expense,
                                                                          tape_pred,
                                                                          upper_bound_output_variables,
                                                                          lower_bound_output_variables)

            unsatisfied_sum_constraint_food_expense, satisfaction_signed_indicator_food_expense, counter_unsatisfaction_sum_food_expense, counter_total_sum_food_expense = \
                check_sum_expression(sum_expression_food_expense,
                                     tf.constant('se', dtype=tf.string),
                                     tf.constant(0, dtype=tf.float32, shape=(batch_size, 1)),
                                     signed_indicator_food_expense,
                                     batch_size,
                                     undo_norm_direction)

            # satisfaction check for the sum constraint income
            sum_expression_income = sum_expression_output_variables(signed_indicator_income,
                                                                    tape_pred,
                                                                    upper_bound_output_variables,
                                                                    lower_bound_output_variables)

            unsatisfied_sum_constraint_income, satisfaction_signed_indicator_income, counter_unsatisfaction_sum_income, counter_total_sum_income = \
                check_sum_expression(sum_expression_income,
                                     tf.constant('se', dtype=tf.string),
                                     tf.math.add(tf.math.multiply(inputs[:, 0],
                                                                  tf.math.subtract(tf.gather_nd(upper_bound_input_variables, (0, 0)),
                                                                                   tf.gather_nd(lower_bound_input_variables, (0, 0)))),
                                                 tf.gather_nd(lower_bound_input_variables, (0, 0))),
                                     signed_indicator_income,
                                     batch_size,
                                     undo_norm_direction)

            # put total number of constraints for each type in a list
            count_con_total = [count_sm_eq, count_sm_st, count_la_eq, count_la_st, counter_total_sum_food_expense, counter_total_sum_income]
            # put total number of unsatisfied constraints for each type in a list
            count_con_unsat = [count_unsat_sm_eq, count_unsat_sm_st, count_unsat_la_eq, count_unsat_la_st,
                               counter_unsatisfaction_sum_food_expense, counter_unsatisfaction_sum_income]

        # put output variables that are not constrained by the specific type to zero as well as the training examples that satisfy the constraint
        unsat_con_sm_eq = tf.math.multiply(tape_pred, ind_sat_sm_eq)
        unsat_con_sm_st = tf.math.multiply(tape_pred, ind_sat_sm_st)
        unsat_con_la_eq = tf.math.multiply(tape_pred, ind_sat_la_eq)
        unsat_con_la_st = tf.math.multiply(tape_pred, ind_sat_la_st)

        # for the unsatisfied sum constraints, perform the sum
        unsatisfied_sum_food_expense = perform_satisfaction_expression(satisfaction_signed_indicator_food_expense, tape_pred)
        unsatisfied_sum_income = perform_satisfaction_expression(satisfaction_signed_indicator_income, tape_pred)

    # compute gradient for the smaller or equal constraints
    grad_sm_eq = tape.gradient(unsat_con_sm_eq, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    # compute gradient for the strictly smaller constraints
    grad_sm_st = tape.gradient(unsat_con_sm_st, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    # compute gradient for the larger or equal constraints
    grad_la_eq = tape.gradient(unsat_con_la_eq, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    # compute gradient for the stirctly larger constraints
    grad_la_st = tape.gradient(unsat_con_la_st, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    # compute gradient for the sum constraint food expense
    grad_sum_food_expense = tape.gradient(unsatisfied_sum_food_expense,
                                          model.trainable_variables,
                                          unconnected_gradients=tf.UnconnectedGradients.ZERO)
    # compute gradient for the sum constraint income
    grad_sum_income = tape.gradient(unsatisfied_sum_income, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)

    del tape  # delete the gradient tape

    sat_ratio = compute_sat_ratio(count_con_unsat, count_con_total)  # compute the satisfaction ratio for the current model (before
    # applying gradients)

    return grad_sm_eq, grad_sm_st, grad_la_eq, grad_la_st, grad_sum_food_expense, grad_sum_income, sat_ratio


def compute_fuzzy_grad(model, inputs, targets, batch_size, output_dim, neur_act, bound_sm_eq, bound_sm_st, bound_la_eq, bound_la_st,
                       ind_sm_eq, ind_sm_st, ind_la_eq, ind_la_st, signed_indicator_food_expense, signed_indicator_income,
                       lower_bound_output_variables, upper_bound_output_variables, lower_bound_input_variables,
                       upper_bound_input_variables, undo_norm_direction):
    """

    Parameters
    ----------
    model : tf.keras.Model
        model that is trained
    inputs : tf.Tensor
        x part of tf.data.Dataset
    targets : tf.Tensor
        y part of tf.data.Dataset
    batch_size : int
        batch-size used during training
    output_dim : int
        output dimension of the network
    neur_act : list[int]
        list of integers denoting the number of neurons or activation functions in the layers of the network.
    bound_sm_eq : tf.Tensor
        consists of all the columns with bounds corresponding to smaller or equal constraints on the output. This tensor has
        dtype=tf.float32 and shape=(batch_size, output_dim).
    bound_sm_st : tf.Tensor
        similarly defined as previous but for the strictly smaller constraints. This tensor has dtype=tf.float32 and
        shape=(batch_size, output_dim).
    bound_la_eq : tf.Tensor
        similarly defined as previous but for the larger or equal constraints. This tensor has dtype=tf.float32 and
        shape=(batch_size, output_dim).
    bound_la_st : tf.Tensor
        similarly defined as previous but for the strictly larger constraints. This tensor has dtype=tf.float32 and
        shape=(batch_size, output_dim).
    ind_sm_eq : tf.Tensor
        corresponding indicator tensor of bound_sm_eq. This tensor has dtype=tf.float32 and shape=(batch_size, output_dim).
    ind_sm_st : tf.Tensor
        similarly defined as ind_sm_eq but for the strictly smaller constraints. This tensor has dtype=tf.float32 and
        shape=(batch_size, output_dim).
    ind_la_eq : tf.Tensor
        similarly defined as ind_sm_eq but for the larger or equal constraints. This tensor has dtype=tf.float32 and
        shape=(batch_size, output_dim).
    ind_la_st : tf.Tensor
        similarly defined as ind_sm_eq but for the strictly larger constraints. This tensor has dtype=tf.float32 and
        shape=(batch_size, output_dim).
    signed_indicator_food_expense : tf.Tensor
        constant tensor consisting of -1, 0 and 1 indicating the presence of an output variable in the constraint and the sign if this
        variable is present. It is assumed that all variables are written on the left-hand side of the smaller or equal sign. If the sum
        constraint would have been strict, then it is assumed that all the variables are written on the left-hand side of the strictly
        smaller sign but then the code used for checking the constraint should also be slightly adjusted.
    signed_indicator_income : tf.Tensor
        constant tensor consisting of -1, 0 and 1 indicating the presence of an output variable in the constraint and the sign if this
        variable is present. It is assumed that all variables are written on the left-hand side of the smaller or equal sign. If the sum
        constraint would have been strict, then it is assumed that all the variables are written on the left-hand side of the strictly
        smaller sign but then the code used for checking the constraint should also be slightly adjusted.
    lower_bound_output_variables : tf.Tensor
        Tensor of shape=(1, output_dimension) and dtype=tf.float32 denoting the lower bound used for the normalization of each output
        variable.
    upper_bound_output_variables : tf.Tensor
        Tensor of shape=(1, output_dimension) and dtype=tf.float32 denoting the upper bound used for the normalization of each output
        variable.
    lower_bound_input_variables : tf.Tensor
        Tensor of shape=(1, input_dimension) and dtype=tf.float32 denoting the lower bound used for the normalization of each input
        variable.
    upper_bound_input_variables : tf.Tensor
        Tensor of shape=(1, input_dimension) and dtype=tf.float32 denoting the upper bound used for the normalization of each input
        variable.
    undo_norm_direction : tf.Tensor
        constant tensor of dtype=tf.float32 and shape(1, output_dimension) containing the factor used to do the normalization of the output
        variables.

    Returns
    ----------
    grad_loss : list[tf.Tensor]
        list of tensors corresponding to the gradient of the loss with respect to the trainable variables of model
    grad_sm_eq : list[tf.Tensor]
        list of tensors corresponding to the gradient of the unsatisfied smaller or equal constraints with respect to the trainable
        variables of model
    grad_sm_st : list[tf.Tensor]
        list of tensors corresponding to the gradient of the unsatisfied strictly smaller constraints with respect to the trainable
        variables of model
    grad_la_eq : list[tf.Tensor]
        list of tensors corresponding to the gradient of the unsatisfied larger or equal constraints with respect to the trainable
        variables of model
    grad_la_st : list[tf.Tensor]
        list of tensors corresponding to the gradient of the unsatisfied strictly larger constraints with respect to the trainable
        variables of model
    grad_sum_food_expense : list[tf.Tensor]
        list of tensors corresponding to the gradient of the unsatisfied sum constraint food expense with respect to the trainable
        variables of model
    grad_sum_income : list[tf.Tensor]
        list of tensors corresponding to the gradient of the unsatisfied sum constraint income with respect to the trainable variables of
        model
    loss_value : tf.Tensor
        value of loss function of the current batch
    metric_value : tf.Tensor
        value of the metric of the current batch
    sat_ratio : tf.Tensor
        satisfaction ratio of the constraints for the current batch

    """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        tape_pred = model(inputs, training=True)  # forward pass to have it recorded by compute_tape
        loss_value = tf.keras.losses.mean_squared_error(targets, tape_pred)

        with tape.stop_recording():
            metric_value = tf.keras.losses.mean_absolute_error(targets,
                                                               tape_pred)  # the metric does not need to be differentiated and thus not necessary to record
            # check satisfaction for smaller or equal constraints
            ind_sat_sm_eq, count_unsat_sm_eq, count_sm_eq = determine_sat_con(tape_pred,
                                                                              bound_sm_eq,
                                                                              batch_size,
                                                                              output_dim,
                                                                              tf.constant('se', dtype=tf.string),
                                                                              ind_sm_eq,
                                                                              ind_sm_st,
                                                                              ind_la_eq,
                                                                              ind_la_st)
            # check satisfaction for strictly smaller constraints
            ind_sat_sm_st, count_unsat_sm_st, count_sm_st = determine_sat_con(tape_pred,
                                                                              bound_sm_st,
                                                                              batch_size,
                                                                              output_dim,
                                                                              tf.constant('ss', dtype=tf.string),
                                                                              ind_sm_eq,
                                                                              ind_sm_st,
                                                                              ind_la_eq,
                                                                              ind_la_st)
            # check satisfaction for larger or equal constraints
            ind_sat_la_eq, count_unsat_la_eq, count_la_eq = determine_sat_con(tape_pred,
                                                                              bound_la_eq,
                                                                              batch_size,
                                                                              output_dim,
                                                                              tf.constant('le', dtype=tf.string),
                                                                              ind_sm_eq,
                                                                              ind_sm_st,
                                                                              ind_la_eq,
                                                                              ind_la_st)
            # check satisfaction for strictly larger constraints
            ind_sat_la_st, count_unsat_la_st, count_la_st = determine_sat_con(tape_pred,
                                                                              bound_la_st,
                                                                              batch_size,
                                                                              output_dim,
                                                                              tf.constant('sl', dtype=tf.string),
                                                                              ind_sm_eq,
                                                                              ind_sm_st,
                                                                              ind_la_eq,
                                                                              ind_la_st)

            #  satisfaction check for the sum constraint food expense
            sum_expression_food_expense = sum_expression_output_variables(signed_indicator_food_expense,
                                                                          tape_pred,
                                                                          upper_bound_output_variables,
                                                                          lower_bound_output_variables)

            unsatisfied_sum_constraint_food_expense, satisfaction_signed_indicator_food_expense, counter_unsatisfaction_sum_food_expense, counter_total_sum_food_expense = \
                check_sum_expression(sum_expression_food_expense,
                                     tf.constant('se', dtype=tf.string),
                                     tf.constant(0, dtype=tf.float32, shape=(batch_size, 1)),
                                     signed_indicator_food_expense,
                                     batch_size,
                                     undo_norm_direction)

            # satisfaction check for the sum constraint income
            sum_expression_income = sum_expression_output_variables(signed_indicator_income,
                                                                    tape_pred,
                                                                    upper_bound_output_variables,
                                                                    lower_bound_output_variables)

            unsatisfied_sum_constraint_income, satisfaction_signed_indicator_income, counter_unsatisfaction_sum_income, counter_total_sum_income = \
                check_sum_expression(sum_expression_income,
                                     tf.constant('se', dtype=tf.string),
                                     tf.math.add(tf.math.multiply(tf.reshape(tf.slice(input_=inputs, begin=[0, 0], size=[-1, 1]), [batch_size, 1]),
                                                                  tf.math.subtract(tf.gather_nd(upper_bound_input_variables, (0, 0)),
                                                                                   tf.gather_nd(lower_bound_input_variables, (0, 0)))),
                                                 tf.gather_nd(lower_bound_input_variables, (0, 0))),
                                     signed_indicator_income,
                                     batch_size,
                                     undo_norm_direction)


            # put total number of constraints for each type in a list
            count_con_total = [count_sm_eq, count_sm_st, count_la_eq, count_la_st, counter_total_sum_food_expense, counter_total_sum_income]
            # put total number of unsatisfied constraints for each type in a list
            count_con_unsat = [count_unsat_sm_eq, count_unsat_sm_st, count_unsat_la_eq, count_unsat_la_st,
                               counter_unsatisfaction_sum_food_expense, counter_unsatisfaction_sum_income]

        loss_smaller_equal = tf.math.reduce_sum(tf.math.reduce_sum(tf.math.maximum(- tape_pred, 0), axis=1))
        loss_larger_equal = tf.math.reduce_sum(tf.math.reduce_sum(tf.math.maximum(tape_pred - 1, 0), axis=1))
        loss_income_constraint = tf.math.reduce_sum(tf.math.maximum(tf.math.reduce_sum((tape_pred * (tf.constant(upper_bound_output_variables, dtype=tf.float32) - tf.constant(lower_bound_output_variables, dtype=tf.float32)) + tf.constant(lower_bound_output_variables, dtype=tf.float32)) * tf.constant([1, 0, 0, 0, 1, 1, 1, 1], dtype=tf.float32), axis=1) - tf.math.reduce_sum(inputs * tf.constant([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32) * (tf.constant(11815988 - 11285, dtype=tf.float32)) + tf.constant([11285, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32), axis=1), 0))/(2188560-1950)
        loss_food_constraint = tf.math.reduce_sum(tf.math.maximum(tf.math.reduce_sum((tape_pred * (tf.constant(upper_bound_output_variables, dtype=tf.float32) - tf.constant(lower_bound_output_variables, dtype=tf.float32)) + tf.constant(lower_bound_output_variables, dtype=tf.float32)) * tf.constant([0, 1, 1, 1, 0, 0, 0, 0], dtype=tf.float32), axis=1) - tf.math.reduce_sum((tape_pred * (tf.constant(upper_bound_output_variables, dtype=tf.float32) - tf.constant(lower_bound_output_variables, dtype=tf.float32)) + tf.constant(lower_bound_output_variables, dtype=tf.float32)) * tf.constant([1, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32), axis=1), 0))/(791848-3704)

        combined_loss = loss_value + loss_smaller_equal + loss_larger_equal + loss_income_constraint + loss_food_constraint

    grad_loss = tape.gradient(combined_loss, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)

    del tape  # delete the gradient tape

    sat_ratio = compute_sat_ratio(count_con_unsat, count_con_total)     # compute the satisfaction ratio for the current model (before
                                                                        # applying gradients)

    return grad_loss, loss_value, metric_value, sat_ratio


def rescale_sum_grad(grad_loss, grad_sm_eq, grad_sm_st, grad_la_eq, grad_la_st, grad_sum_food_expense, grad_sum_income, neur_act,
                     rescale_constant):
    """Rescale gradients of constraints according to the gradient of the loss. In addition, the rescaled gradients are all combined into a
    single gradient that needs to be applied.

    The rescaling is done over the incoming weight nodes and for each activation function separately.

    Parameters
    ----------
    grad_loss : list[tf.Tensor]
        list of tensors corresponding to the gradient of the loss with respect to the trainable variables of model
    grad_sm_eq : list[tf.Tensor]
        list of tensors corresponding to the gradient of the unsatisfied smaller or equal constraints with respect to the trainable
        variables of model
    grad_sm_st : list[tf.Tensor]
        list of tensors corresponding to the gradient of the unsatisfied strictly smaller constraints with respect to the trainable
        variables of model
    grad_la_eq : list[tf.Tensor]
        list of tensors corresponding to the gradient of the unsatisfied larger or equal constraints with respect to the trainable
        variables of model
    grad_la_st : list[tf.Tensor]
        list of tensors corresponding to the gradient of the unsatisfied strictly larger constraints with respect to the trainable
        variables of model
    grad_sum_food_expense : list[tf.Tensor]
        List of tensors corresponding to the gradient of the unsatisfied sum constraint food expense with respect to the trainable
        variables of model.
    grad_sum_income : list[tf.Tensor]
        List of tensors corresponding to the gradient of the unsatisfied sum constraint income with respect to the trainable variables of
        model.
    neur_act : list[int]
        list of integers denoting the number of neurons or activation functions in the layers of the network.
    rescale_constant : tf.Tensor
        tensor of dtype=tf.float32 and shape=(,) indicating the factor with which the normalized gradients need to be multiplied in
        addition to the maximum of the loss function.

    Returns
    ----------
    scale_grad : list[tf.Tensor]
        list of tensors corresponding to the gradient of loss and constraints combined with respect to the trainable variables of model.
    max_layers : list[tf.Tensor]
        list of tensors corresponding to the maxima of the absolute value of the gradient of the loss function in the current batch.

    """
    # add together the smaller and larger constraints respectively. This is allowed because they will have gradients of the same sign.
    # In addition, it improves the scalability of the method.
    grad_sm = []
    grad_la = []
    for i in range(0, len(neur_act)):
        grad_sm.append(tf.math.add(grad_sm_eq[i], grad_sm_st[i]))
        grad_la.append(tf.math.add(grad_la_eq[i], grad_la_st[i]))

    # determine the L2-norm of each gradient of smaller constraints (for now it doesn't make much sense to do this instead of just
    # taking absolute value, but when we want to take L2-norm along a given layer, then this code should generalize with minimal
    # adjustments)
    norm_grad_sm = []  # initialize list for normalizations
    norm_grad_la = []
    norm_grad_sum_food_expense = []
    norm_grad_sum_income = []
    max_layers = []  # initialize list for the maximization
    for i in range(0, len(neur_act)):
        if i % 2 == 0:  # this correspond to a weight matrix
            norm_grad_sm.append(tf.math.l2_normalize(grad_sm[i], axis=1))
            norm_grad_la.append(tf.math.l2_normalize(grad_la[i], axis=1))
            # normalization of gradients of sum constraint
            norm_grad_sum_food_expense.append(tf.math.l2_normalize(grad_sum_food_expense[i], axis=1))
            norm_grad_sum_income.append(tf.math.l2_normalize(grad_sum_income[i], axis=1))

            layer_max = tf.cast(tf.norm(tf.math.abs(grad_loss[i]), axis=1), dtype=tf.float32)
            scale_layer_max = tf.math.scalar_mul(rescale_constant, layer_max)  # rescale with factor 2
            max_layers.append(scale_layer_max)

        else:  # this corresponds to activation functions
            sign_grad_sm = tf.cast(tf.math.sign(grad_sm[i]), dtype=tf.float32)
            norm_grad_sm.append(sign_grad_sm)
            sign_grad_la = tf.cast(tf.math.sign(grad_la[i]), dtype=tf.float32)
            norm_grad_la.append(sign_grad_la)
            sign_grad_sum_food_expense = tf.cast(tf.math.sign(grad_sum_food_expense[i]), dtype=tf.float32)
            norm_grad_sum_food_expense.append(sign_grad_sum_food_expense)
            sign_grad_sum_income = tf.cast(tf.math.sign(grad_sum_income[i]), dtype=tf.float32)
            norm_grad_sum_income.append(sign_grad_sum_income)

            max_layers.append(tf.math.scalar_mul(rescale_constant, tf.math.abs(grad_loss[i])))

    # rescale normalized gradients smaller constraints
    scale_grad_sm = []  # initialize list for rescaling
    scale_grad_la = []
    scale_grad_sum_food_expense = []
    scale_grad_sum_income = []
    for i in range(0, len(neur_act)):
        if i % 2 == 0:
            resh_max_layer = tf.reshape(max_layers[i], [neur_act[i], 1])
            scale_grad_sm.append(tf.math.multiply(resh_max_layer, norm_grad_sm[i]))
            scale_grad_la.append(tf.math.multiply(resh_max_layer, norm_grad_la[i]))
            scale_grad_sum_food_expense.append(tf.math.multiply(resh_max_layer, norm_grad_sum_food_expense[i]))
            scale_grad_sum_income.append(tf.math.multiply(resh_max_layer, norm_grad_sum_income[i]))
        else:
            scale_grad_sm.append(tf.math.multiply(max_layers[i], norm_grad_sm[i]))
            scale_grad_la.append(tf.math.multiply(max_layers[i], norm_grad_la[i]))
            scale_grad_sum_food_expense.append(tf.math.multiply(max_layers[i], norm_grad_sum_food_expense[i]))
            scale_grad_sum_income.append(tf.math.multiply(max_layers[i], norm_grad_sum_income[i]))

    # add all the gradients together (with the corresponding signs)
    scale_grad = []  # initialize list for all gradients combined
    for i in range(0, len(neur_act)):
        temp_grad = tf.math.subtract(tf.math.add_n([grad_loss[i],
                                                    scale_grad_sm[i],
                                                    scale_grad_sum_food_expense[i],
                                                    scale_grad_sum_income[i]]),
                                     scale_grad_la[i])
        if i % 2 == 0:
            temp_grad = tf.reshape(temp_grad, [neur_act[i], neur_act[i+1]])
        else:
            temp_grad = tf.reshape(temp_grad, [neur_act[i], ])
        scale_grad.append(temp_grad)

    return scale_grad, max_layers


def unsup_rescale_sum_grad(grad_sm_eq, grad_sm_st, grad_la_eq, grad_la_st, grad_sum_food_expense, grad_sum_income, neur_act, max_layers):
    """Rescale gradients of constraints according to the gradient of the loss. In addition, the rescaled gradients are all combined into a
    single gradient that needs to be applied.

    The rescaling is done over the incoming weight nodes and for each activation function separately.

    Parameters
    ----------
    grad_sm_eq : list[tf.Tensor]
        list of tensors corresponding to the gradient of the unsatisfied smaller or equal constraints with respect to the trainable
        variables of model
    grad_sm_st : list[tf.Tensor]
        list of tensors corresponding to the gradient of the unsatisfied strictly smaller constraints with respect to the trainable
        variables of model
    grad_la_eq : list[tf.Tensor]
        list of tensors corresponding to the gradient of the unsatisfied larger or equal constraints with respect to the trainable
        variables of model
    grad_la_st : list[tf.Tensor]
        list of tensors corresponding to the gradient of the unsatisfied strictly larger constraints with respect to the trainable
        variables of model
    grad_sum_food_expense : list[tf.Tensor]
        List of tensors corresponding to the gradient of the unsatisfied sum constraint food expense with respect to the trainable
        variables of model.
    grad_sum_income : list[tf.Tensor]
        List of tensors corresponding to the gradient of the unsatisfied sum constraint income with respect to the trainable variables of
        model.
    neur_act : list[int]
        list of integers denoting the number of neurons or activation functions in the layers of the network.
    max_layers : list[tf.Tensor]
        list of tensors corresponding to the maxima of the absolute value of the gradient of the loss function in the previous supervised batch.

    Returns
    ----------
    scale_grad : list[tf.Tensor]
        list of tensors corresponding to the gradient of loss and constraints combined with respect to the trainable variables of model.

    """
    # add together the smaller and larger constraints respectively. This is allowed because they will have gradients of the same sign.
    # In addition, it improves the scalability of the method.
    grad_sm = []
    grad_la = []
    for i in range(0, len(neur_act)):
        grad_sm.append(tf.math.add(grad_sm_eq[i], grad_sm_st[i]))
        grad_la.append(tf.math.add(grad_la_eq[i], grad_la_st[i]))

    # determine the L2-norm of each gradient of smaller constraints (for now it doesn't make much sense to do this instead of just
    # taking absolute value, but when we want to take L2-norm along a given layer, then this code should generalize with minimal
    # adjustments)
    norm_grad_sm = []  # initialize list for normalizations
    norm_grad_la = []
    norm_grad_sum_food_expense = []
    norm_grad_sum_income = []
    for i in range(0, len(neur_act)):
        if i % 2 == 0:  # this correspond to a weight matrix
            norm_grad_sm.append(tf.math.l2_normalize(grad_sm[i], axis=1))
            norm_grad_la.append(tf.math.l2_normalize(grad_la[i], axis=1))
            # normalization of gradients of sum constraint
            norm_grad_sum_food_expense.append(tf.math.l2_normalize(grad_sum_food_expense[i], axis=1))
            norm_grad_sum_income.append(tf.math.l2_normalize(grad_sum_income[i], axis=1))

        else:  # this corresponds to activation functions
            sign_grad_sm = tf.cast(tf.math.sign(grad_sm[i]), dtype=tf.float32)
            norm_grad_sm.append(sign_grad_sm)
            sign_grad_la = tf.cast(tf.math.sign(grad_la[i]), dtype=tf.float32)
            norm_grad_la.append(sign_grad_la)
            sign_grad_sum_food_expense = tf.cast(tf.math.sign(grad_sum_food_expense[i]), dtype=tf.float32)
            norm_grad_sum_food_expense.append(sign_grad_sum_food_expense)
            sign_grad_sum_income = tf.cast(tf.math.sign(grad_sum_income[i]), dtype=tf.float32)
            norm_grad_sum_income.append(sign_grad_sum_income)

    # rescale normalized gradients smaller constraints
    scale_grad_sm = []  # initialize list for rescaling
    scale_grad_la = []
    scale_grad_sum_food_expense = []
    scale_grad_sum_income = []
    for i in range(0, len(neur_act)):
        if i % 2 == 0:
            resh_max_layer = tf.reshape(max_layers[i], [neur_act[i], 1])

            scale_grad_sm.append(tf.math.multiply(resh_max_layer, norm_grad_sm[i]))
            scale_grad_la.append(tf.math.multiply(resh_max_layer, norm_grad_la[i]))
            scale_grad_sum_food_expense.append(tf.math.multiply(resh_max_layer, norm_grad_sum_food_expense[i]))
            scale_grad_sum_income.append(tf.math.multiply(resh_max_layer, norm_grad_sum_income[i]))
        else:
            scale_grad_sm.append(tf.math.multiply(max_layers[i], norm_grad_sm[i]))
            scale_grad_la.append(tf.math.multiply(max_layers[i], norm_grad_la[i]))
            scale_grad_sum_food_expense.append(tf.math.multiply(max_layers[i], norm_grad_sum_food_expense[i]))
            scale_grad_sum_income.append(tf.math.multiply(max_layers[i], norm_grad_sum_income[i]))

    # add all the gradients together (with the corresponding signs)
    scale_grad = []  # initialize list for all gradients combined
    for i in range(0, len(neur_act)):
        temp_grad = tf.math.subtract(tf.math.add_n([scale_grad_sm[i],
                                                    scale_grad_sum_food_expense[i],
                                                    scale_grad_sum_income[i]]),
                                     scale_grad_la[i])
        if i % 2 == 0:
            temp_grad = tf.reshape(temp_grad, [neur_act[i], neur_act[i + 1]])
        else:
            temp_grad = tf.reshape(temp_grad, [neur_act[i], ])
        scale_grad.append(temp_grad)

    return scale_grad


def initialize_indicator_tensors(batch_size, output_dim, neur_act, list_bound_constraints):
    """Initialize all the constant tensors that are necessary as inputs for the training procedure since they are invariant during training.

    This function uses the functions construct_bound_con, sum_output_variables, expression1_convert_bound_con and
    expression2_convert_bound_con. For more information about these functions, please read the documentation of these function themselves.

    Parameters
    ----------
    batch_size : int
        integer denoting the batch-size used for the dataset.
    output_dim : int
        output dimension of the network
    neur_act : list[int]
        list containing the first dimension of the weight matrices and the number of activation functions.
    list_bound_constraints : list[list[tf.Tensor]]
        list containing the bound constraints.

    Returns
    ----------
    input_tensor : list[tf.Tensor]
        list containing all the output tensors

    """
    if list_bound_constraints is None or list_bound_constraints == []:
        return []
    else:
        bound_se, bound_ss, bound_le, bound_ls, bound_ind_se, bound_ind_ss, bound_ind_le, bound_ind_ls = \
            construct_bound_con(list_bound_constraints, batch_size, output_dim)

        # the maximum air temperature is assumed to be output variable with index 0
        # the minimum air temperature is assumed to be output variable with index 1
        # the constraint food expense
        sign_ind_food_expense = sum_output_variables(index_positive_output=[tf.constant(1, dtype=tf.int32),
                                                                            tf.constant(2, dtype=tf.int32),
                                                                            tf.constant(3, dtype=tf.int32)],
                                                     index_negative_output=[tf.constant(0, dtype=tf.int32)],
                                                     batch_size=batch_size,
                                                     output_dim=output_dim)

        # the constraint income
        sign_ind_income = sum_output_variables(index_positive_output=[tf.constant(0, dtype=tf.int32), tf.constant(4, dtype=tf.int32),
                                                                      tf.constant(5, dtype=tf.int32), tf.constant(6, dtype=tf.int32),
                                                                      tf.constant(7, dtype=tf.int32)],
                                               index_negative_output=[],
                                               batch_size=batch_size,
                                               output_dim=output_dim)

        input_tensor = [bound_se, bound_ss, bound_le, bound_ls, bound_ind_se, bound_ind_ss, bound_ind_le, bound_ind_ls,
                        sign_ind_food_expense, sign_ind_income]

        return input_tensor


# The code below here is used the evaluate the model on a validation set.
def validation_check(model, inputs, targets, batch_size, output_dim, neur_act, bound_sm_eq, bound_sm_st, bound_la_eq, bound_la_st,
                     ind_sm_eq, ind_sm_st, ind_la_eq, ind_la_st, signed_indicator_food_expense, signed_indicator_income,
                     lower_bound_output_variables, upper_bound_output_variables, lower_bound_input_variables, upper_bound_input_variables,
                     undo_norm_direction):
    """Compute the satisfaction ratio for the validation set.

    Observe that this function is in fact what happens in the computation of the gradient in the part that is not recorded. This code could
    be somewhat optimized in the sense that strictly speaking this piece of code computes slightly too much.

    Parameters
    ----------
    model : tf.keras.Model
        model that is trained
    inputs : tf.Tensor
        x part of tf.data.Dataset
    targets : tf.Tensor
        y part of tf.data.Dataset
    batch_size : int
        batch-size used during training
    output_dim : int
        output dimension of the network
    neur_act : list[int]
        list of integers denoting the number of neurons and activation functions in the layers of the network.
    bound_sm_eq : tf.Tensor
        consists of all the columns with bounds corresponding to smaller or equal constraints on the output. This tensor has
        dtype=tf.float32 and shape=(batch_size, output_dim).
    bound_sm_st : tf.Tensor
        similarly defined as previous but for the strictly smaller constraints. This tensor has dtype=tf.float32 and
        shape=(batch_size, output_dim).
    bound_la_eq : tf.Tensor
        similarly defined as previous but for the larger or equal constraints. This tensor has dtype=tf.float32 and
        shape=(batch_size, output_dim).
    bound_la_st : tf.Tensor
        similarly defined as previous but for the strictly larger constraints. This tensor has dtype=tf.float32 and
        shape=(batch_size, output_dim).
    ind_sm_eq : tf.Tensor
        corresponding indicator tensor of bound_sm_eq. This tensor has dtype=tf.float32 and shape=(batch_size, output_dim).
    ind_sm_st : tf.Tensor
        similarly defined as ind_sm_eq but for the strictly smaller constraints. This tensor has dtype=tf.float32 and
        shape=(batch_size, output_dim).
    ind_la_eq : tf.Tensor
        similarly defined as ind_sm_eq but for the larger or equal constraints. This tensor has dtype=tf.float32 and
        shape=(batch_size, output_dim).
    ind_la_st : tf.Tensor
        similarly defined as ind_sm_eq but for the strictly larger constraints. This tensor has dtype=tf.float32 and
        shape=(batch_size, output_dim).
    signed_indicator_food_expense : tf.Tensor
        constant tensor consisting of -1, 0 and 1 indicating the presence of an output variable in the constraint and the sign if this
        variable is present. It is assumed that all variables are written on the left-hand side of the smaller or equal sign. If the sum
        constraint would have been strict, then it is assumed that all the variables are written on the left-hand side of the strictly
        smaller sign but then the code used for checking the constraint should also be slightly adjusted.
    signed_indicator_income : tf.Tensor
        constant tensor consisting of -1, 0 and 1 indicating the presence of an output variable in the constraint and the sign if this
        variable is present. It is assumed that all variables are written on the left-hand side of the smaller or equal sign. If the sum
        constraint would have been strict, then it is assumed that all the variables are written on the left-hand side of the strictly
        smaller sign but then the code used for checking the constraint should also be slightly adjusted.
    lower_bound_output_variables : tf.Tensor
        Tensor of shape=(1, output_dimension) and dtype=tf.float32 denoting the lower bound used for the normalization of each output
        variable.
    upper_bound_output_variables : tf.Tensor
        Tensor of shape=(1, output_dimension) and dtype=tf.float32 denoting the upper bound used for the normalization of each output
        variable.
    lower_bound_input_variables : tf.Tensor
        Tensor of shape=(1, input_dimension) and dtype=tf.float32 denoting the lower bound used for the normalization of each input
        variable.
    upper_bound_input_variables : tf.Tensor
        Tensor of shape=(1, input_dimension) and dtype=tf.float32 denoting the upper bound used for the normalization of each input
        variable.
    undo_norm_direction : tf.Tensor
        constant tensor of dtype=tf.float32 and shape(1, output_dimension) containing the factor used to do the normalization of the output
        variables.

    Returns
    ----------
    loss_validation : tf.Tensor
        constant tensor containing the loss value on the batch in the validation set.
    metric_validation : tf.Tensor
        constant tensor containing the metric value on the batch in the validation set.
    satisfaction_ratio_val : tf.Tensor
        constant tensor containing the satisfaction ratio on the batch in the validation set.

    """
    predictions = model(inputs, training=False)
    #loss_validation = loss_function(targets, predictions)
    loss_validation = tf.keras.losses.mean_squared_error(targets, predictions)
    #metric_validation = metric_function(targets, predictions)
    metric_validation = tf.keras.losses.mean_absolute_error(targets, predictions)

    ind_sat_sm_eq, count_unsat_sm_eq, count_sm_eq = determine_sat_con(predictions,
                                                                      bound_sm_eq,
                                                                      batch_size,
                                                                      output_dim,
                                                                      tf.constant('se', dtype=tf.string),
                                                                      ind_sm_eq,
                                                                      ind_sm_st,
                                                                      ind_la_eq,
                                                                      ind_la_st)
    ind_sat_sm_st, count_unsat_sm_st, count_sm_st = determine_sat_con(predictions,
                                                                      bound_sm_st,
                                                                      batch_size,
                                                                      output_dim,
                                                                      tf.constant('ss', dtype=tf.string),
                                                                      ind_sm_eq,
                                                                      ind_sm_st,
                                                                      ind_la_eq,
                                                                      ind_la_st)
    ind_sat_la_eq, count_unsat_la_eq, count_la_eq = determine_sat_con(predictions,
                                                                      bound_la_eq,
                                                                      batch_size,
                                                                      output_dim,
                                                                      tf.constant('le', dtype=tf.string),
                                                                      ind_sm_eq,
                                                                      ind_sm_st,
                                                                      ind_la_eq,
                                                                      ind_la_st)
    ind_sat_la_st, count_unsat_la_st, count_la_st = determine_sat_con(predictions,
                                                                      bound_la_st,
                                                                      batch_size,
                                                                      output_dim,
                                                                      tf.constant('sl', dtype=tf.string),
                                                                      ind_sm_eq,
                                                                      ind_sm_st,
                                                                      ind_la_eq,
                                                                      ind_la_st)

    # satisfaction check for the sum constraint food expense
    sum_expression_food_expense = sum_expression_output_variables(signed_indicator_food_expense,
                                                                  predictions,
                                                                  upper_bound_output_variables,
                                                                  lower_bound_output_variables)

    unsatisfied_sum_constraint_food_expense, satisfaction_signed_indicator_food_expense, counter_unsatisfaction_sum_food_expense, counter_total_sum_food_expense = \
         check_sum_expression(sum_expression_food_expense,
                              tf.constant('se', dtype=tf.string),
                              tf.constant(0, dtype=tf.float32, shape=(batch_size, 1)),
                              signed_indicator_food_expense,
                              batch_size,
                              undo_norm_direction)

    # satisfaction check for the sum constraint income
    sum_expression_income = sum_expression_output_variables(signed_indicator_income,
                                                            predictions,
                                                            upper_bound_output_variables,
                                                            lower_bound_output_variables)

    unsatisfied_sum_constraint_income, satisfaction_signed_indicator_income, counter_unsatisfaction_sum_income, counter_total_sum_income = \
        check_sum_expression(sum_expression_income,
                             tf.constant('se', dtype=tf.string),
                             tf.math.add(tf.math.multiply(inputs[:, 0],
                                                          tf.math.subtract(tf.gather_nd(upper_bound_input_variables, (0, 0)),
                                                                           tf.gather_nd(lower_bound_input_variables, (0, 0)))),
                                         tf.gather_nd(lower_bound_input_variables, (0, 0))),
                             signed_indicator_income,
                             batch_size,
                             undo_norm_direction)

    # put total number of constraints for each type in a list
    count_con_total = [count_sm_eq, count_sm_st, count_la_eq, count_la_st, counter_total_sum_food_expense, counter_total_sum_income]
    # put total number of unsatisfied constraints for each type in a list
    count_con_unsat = [count_unsat_sm_eq, count_unsat_sm_st, count_unsat_la_eq, count_unsat_la_st, counter_unsatisfaction_sum_food_expense,
                       counter_unsatisfaction_sum_income]

    satisfaction_ratio_val = compute_sat_ratio(count_con_unsat, count_con_total)

    return loss_validation, metric_validation, satisfaction_ratio_val


# The code below is the training procedure used for training the model with all constraints.
def train_model(model, input_dim, output_dim, optimizer, neur_act, list_bound_constraints, train_number, train_dataset, train_batch_size,
                val_number, val_dataset, val_batch_size, max_epochs, max_con_epochs, rescale_constant, learning_rate, acceptable_sat_ratio,
                max_bumps, max_time_after_bump, max_constant_perf, max_restarts, save_location_model, save_file_model,
                lower_bound_output_variables, upper_bound_output_variables, lower_bound_input_variables,
                upper_bound_input_variables):
    """Train the given model.

    Parameters
    ----------
    model : tf.keras.Model
        model to be trained.
    input_dim : int
        integer denoting the input dimension of the network.
    output_dim : int
        integer denoting in the output dimension of the network.
    optimizer : tf.keras.Optimizer
        optimizer used during training.
    neur_act : list[int]
        list containing the first dimension of the weight matrices and the number of activation functions for each hidden layer.
    list_bound_constraints : list[list[tf.Tensor]]
        list containing the bound cosntraints.
    train_number : int
        integer denoting the number of training examples.
    train_dataset : tf.data.Dataset
        dataset containing the training examples.
    train_batch_size : int
        batch-size used for the training set.
    val_number : int
        integer denoting the number of validation examples.
    val_dataset : tf.data.Dataset
        dataset containing the validation examples.
    val_batch_size : int
        batch-size used for the validation set.
    max_epochs : int
        integer denoting the maximum number of epochs that can be used to optimize the loss function alone.
    max_con_epochs : int
        integer denoting the maximum number of epochs that can be used to optimize the constraints.
    rescale_constant : float
        factor to rescale the normalized gradients before multiplying with sup norm of gradients of loss function
    learning_rate : float
        float denoting the initial value of the learning rate of the optimizer.
    acceptable_sat_ratio : float
        float in [0,1] indicating how many constraints need to be satisfied in order to lead to an acceptable model. Typically, one wants
        to set this to 1 but for highly constrained problems one can relax this to find an approximation of a model satisfying all the
        constraints.
    max_bumps : int
        maximum number of bumps without improving the current best model that is allowed before resetting the training to the best model.
    max_time_after_bump : int
        maximum number of epochs after a single bump in which there is no improvement to the best model.
    max_constant_perf : int
        maximum number of epochs in which there is no improve in performance compared to the best and all constraints reached an acceptable
        satisfaction ratio. In other words, the maximum number of epochs in which the model does not improve and there is no bump. This acts
        in the same way as early-stopping would impact the training procedure.
    max_restarts : int
        maximum number of restarts that can be performed during training.
    save_location_model : string
        path to file for saving the model weights of the best performing model during training.
    save_file_model : string
        Name of file containing the weights of the trained model.
    lower_bound_output_variables : tf.Tensor
        Tensor of shape=(1, output_dimension) and dtype=tf.float32 denoting the lower bound used for the normalization of each output
        variable.
    upper_bound_output_variables : tf.Tensor
        Tensor of shape=(1, output_dimension) and dtype=tf.float32 denoting the upper bound used for the normalization of each output
        variable.
    lower_bound_input_variables : tf.Tensor
        Tensor of shape=(1, input_dimension) and dtype=tf.float32 denoting the lower bound used for the normalization of each input
        variable.
    upper_bound_input_variables : tf.Tensor
        Tensor of shape=(1, input_dimension) and dtype=tf.float32 denoting the upper bound used for the normalization of each input
        variable.

    Returns
    ----------
    model : tf.keras.Model
        best performing model obtained during training. This is not necessarily the last model obtained during training model.
    train_loss_results : list[tf.Tensor]
        List containing the value of the loss function of the training set for each epoch.
    train_metric_results : list[tf.Tensor]
        List containing the value of the metric of the training set for each epoch.
    train_satisfaction_results : list[tf.Tensor]
        List containing the satisfaction ratio of the training set for each epoch.
    val_loss_results : list[tf.Tensor]
        List containing the value of the loss function of the validation set for each epoch.
    val_metric_results : list[tf.Tensor]
        List containing the value of the metric of the validation set for each epoch.
    val_satisfaction_results : list[tf.Tensor]
        List containing the satisfaction ratio of the validation set for each epoch.
    timing : list[float]
        List containing the end time of each epoch.

    """
    ### initialize constant tensors that are invariant for the training process
    # tensors for train set
    ind_tensors_train = initialize_indicator_tensors(batch_size=train_batch_size,
                                                     output_dim=output_dim,
                                                     neur_act=neur_act,
                                                     list_bound_constraints=list_bound_constraints)

    if not train_number % train_batch_size == 0:  # for the final batch that is smaller than the others, the constant tensors containing
        # bounds and indicators need to be smaller
        new_train_batch_size = train_number % train_batch_size
        ind_tensors_train_last = initialize_indicator_tensors(batch_size=new_train_batch_size,
                                                              output_dim=output_dim,
                                                              neur_act=neur_act,
                                                              list_bound_constraints=list_bound_constraints)

    # tensors for validation set (only when the user gave a validation set)
    ind_tensors_val = initialize_indicator_tensors(batch_size=val_batch_size,
                                                   output_dim=output_dim,
                                                   neur_act=neur_act,
                                                   list_bound_constraints=list_bound_constraints)

    if not val_number % val_batch_size == 0:
        new_val_batch_size = val_number % val_batch_size
        ind_tensors_val_last = initialize_indicator_tensors(batch_size=new_val_batch_size,
                                                            output_dim=output_dim,
                                                            neur_act=neur_act,
                                                            list_bound_constraints=list_bound_constraints)

    lower_bound_output_variables = tf.reshape(tf.convert_to_tensor(lower_bound_output_variables), [1, output_dim])
    upper_bound_output_variables = tf.reshape(tf.convert_to_tensor(upper_bound_output_variables), [1, output_dim])
    lower_bound_input_variables = tf.reshape(tf.convert_to_tensor(lower_bound_input_variables), [1, 1])
    upper_bound_input_variables = tf.reshape(tf.convert_to_tensor(upper_bound_input_variables), [1, 1])
    undo_norm_direction = tf.math.subtract(upper_bound_output_variables, lower_bound_output_variables)

    ### define variables that are used throughout the training procedure
    # define number of bound constraints
    length_list_bound_constraints = len(list_bound_constraints)

    # initialize lists for recording results
    train_loss_results = []
    train_metric_results = []
    train_satisfaction_results = []

    val_loss_results = []
    val_metric_results = []
    val_satisfaction_results = []

    # initialize counters for epoch
    epoch_constraints = 0
    epoch = 1

    # define learning_rates
    learning_rate_loss = learning_rate
    list_lr_loss = [learning_rate_loss]
    # learning_rate_loss = 0.005
    learning_rate_constraints = learning_rate 
    list_lr_con = [learning_rate_constraints]
    list_lr = [learning_rate]
    new_learning_rate = learning_rate

    best_learning_rate = learning_rate

    # initialize metrics for recording values of loss function, additional metrics and satisfaction ratio for each batch in an epoch
    epoch_train_loss_average = tf.keras.metrics.Mean()
    epoch_train_metric_average = tf.keras.metrics.Mean()
    epoch_train_satisfaction_average = tf.keras.metrics.Mean()

    epoch_val_loss_average = tf.keras.metrics.Mean()
    epoch_val_metric_average = tf.keras.metrics.Mean()
    epoch_val_satisfaction_average = tf.keras.metrics.Mean()

    # initialize previous values of loss function, additional metrics and satisfaction ratio
    prev_train_sat_average = 10**3
    prev_train_loss_average = 10**3
    prev_train_met_average = 10**3

    prev_val_sat_average = 10**3
    prev_val_loss_average = 10**3
    prev_val_met_average = 10**3

    best_loss_average = 10**3
    best_metric_average = 10**3

    # initialize counter for training steps that did not change the network's performance
    constant_performance_val = 1
    counter_optimize_constraints = 1
    times_reduced_constraints = 0
    first_time_loss = 2
    first_time_con = 1

    # determine number of batches for rescale factor
    number_batches = np.ceil(train_number / train_batch_size)
    number_batches_1 = tf.math.add(number_batches, tf.constant(1, dtype=tf.float32))

    # initialize counters for stopping criterion
    number_bumps = 0  # counter for the number of bumps in this training procedure (without resetting to best model)
    number_resets = 0  # counter for the number of resets to the best model
    number_bump_con = 0  # counter for number of epochs where there is a constant (or increasing) performance after a bump
    number_con = 0  # counter for number of epochs where there is no improvement of performance (and no bumps)

    epoch_resets = []  # initialize list to collect the epochs where there was a reset
    epoch_bumps = []  # initialize list to collect the epochs where there was a bump

    timing = []  # initialize empty list for the timing of the epochs of the training

    timing.append(time.time())

    saved_model = False

    ### start of training procedure
    while epoch - epoch_constraints < max_epochs and epoch_constraints < max_con_epochs:
        # initialize/reset counter unsatisfied batches
        unsat_batch = tf.constant(0, dtype=tf.float32)
        changed_learning_rate = False
        for x, y in train_dataset:
            current_batch_size = np.size(x, 0)  # determine the size of the batch
            if current_batch_size == train_batch_size:
                # optimize model
                gradient_loss, gradient_smaller_equal, gradient_smaller_strict, gradient_larger_equal, gradient_larger_strict, gradient_sum_constraint_food_expense, gradient_sum_constraint_income, train_loss_value, train_metric_value, train_satisfaction_ratio \
                    = compute_sum_grad_undo_norm(model=model,
                                                 inputs=x,
                                                 targets=y,
                                                 batch_size=current_batch_size,
                                                 output_dim=output_dim,
                                                 neur_act=neur_act,
                                                 bound_sm_eq=ind_tensors_train[0],
                                                 bound_sm_st=ind_tensors_train[1],
                                                 bound_la_eq=ind_tensors_train[2],
                                                 bound_la_st=ind_tensors_train[3],
                                                 ind_sm_eq=ind_tensors_train[4],
                                                 ind_sm_st=ind_tensors_train[5],
                                                 ind_la_eq=ind_tensors_train[6],
                                                 ind_la_st=ind_tensors_train[7],
                                                 signed_indicator_food_expense=ind_tensors_train[8],
                                                 signed_indicator_income=ind_tensors_train[9],
                                                 lower_bound_output_variables=lower_bound_output_variables,
                                                 upper_bound_output_variables=upper_bound_output_variables,
                                                 lower_bound_input_variables=lower_bound_input_variables,
                                                 upper_bound_input_variables=upper_bound_input_variables,
                                                 undo_norm_direction=undo_norm_direction)

                if not train_satisfaction_ratio == 1:
                    unsat_batch = tf.math.add(unsat_batch, tf.constant(1, dtype=tf.float32))

                rescaled_gradients, max_layers = rescale_sum_grad(grad_loss=gradient_loss,
                                                                  grad_sm_eq=gradient_smaller_equal,
                                                                  grad_sm_st=gradient_smaller_strict,
                                                                  grad_la_eq=gradient_larger_equal,
                                                                  grad_la_st=gradient_larger_strict,
                                                                  grad_sum_food_expense=gradient_sum_constraint_food_expense,
                                                                  grad_sum_income=gradient_sum_constraint_income,
                                                                  neur_act=neur_act,
                                                                  rescale_constant=rescale_constant)

                optimizer.apply_gradients(zip(rescaled_gradients, model.trainable_variables))
            else:  # use the constant tensors and indicator tensors with size corresponding to the last (and smaller) batch
                # optimize model
                gradient_loss, gradient_smaller_equal, gradient_smaller_strict, gradient_larger_equal, gradient_larger_strict, gradient_sum_constraint_food_expense, gradient_sum_constraint_income, train_loss_value, train_metric_value, train_satisfaction_ratio \
                    = compute_sum_grad_undo_norm(model=model,
                                                 inputs=x,
                                                 targets=y,
                                                 batch_size=current_batch_size,
                                                 output_dim=output_dim,
                                                 neur_act=neur_act,
                                                 bound_sm_eq=ind_tensors_train_last[0],
                                                 bound_sm_st=ind_tensors_train_last[1],
                                                 bound_la_eq=ind_tensors_train_last[2],
                                                 bound_la_st=ind_tensors_train_last[3],
                                                 ind_sm_eq=ind_tensors_train_last[4],
                                                 ind_sm_st=ind_tensors_train_last[5],
                                                 ind_la_eq=ind_tensors_train_last[6],
                                                 ind_la_st=ind_tensors_train_last[7],
                                                 signed_indicator_food_expense=ind_tensors_train_last[8],
                                                 signed_indicator_income=ind_tensors_train_last[9],
                                                 lower_bound_output_variables=lower_bound_output_variables,
                                                 upper_bound_output_variables=upper_bound_output_variables,
                                                 lower_bound_input_variables=lower_bound_input_variables,
                                                 upper_bound_input_variables=upper_bound_input_variables,
                                                 undo_norm_direction=undo_norm_direction)

                rescaled_gradients, max_layers = rescale_sum_grad(grad_loss=gradient_loss,
                                                                  grad_sm_eq=gradient_smaller_equal,
                                                                  grad_sm_st=gradient_smaller_strict,
                                                                  grad_la_eq=gradient_larger_equal,
                                                                  grad_la_st=gradient_larger_strict,
                                                                  grad_sum_food_expense=gradient_sum_constraint_food_expense,
                                                                  grad_sum_income=gradient_sum_constraint_income,
                                                                  neur_act=neur_act,
                                                                  rescale_constant=rescale_constant)

                optimizer.apply_gradients(zip(rescaled_gradients, model.trainable_variables))

            # track progress from training batch
            epoch_train_loss_average.update_state(values=train_loss_value, sample_weight=current_batch_size / train_batch_size)
            epoch_train_metric_average.update_state(values=train_metric_value, sample_weight=current_batch_size / train_batch_size)
            epoch_train_satisfaction_average.update_state(values=train_satisfaction_ratio, sample_weight=current_batch_size / train_batch_size)

        rescale_constant = tf.math.multiply(tf.math.subtract(number_batches_1, unsat_batch), tf.constant(2, dtype=tf.float32))

        # track progress from epoch
        train_loss_results.append(epoch_train_loss_average.result())
        train_metric_results.append(epoch_train_metric_average.result())
        train_satisfaction_results.append(epoch_train_satisfaction_average.result())

        # check performance on validation set
        for x, y in val_dataset:
            current_batch_size = np.size(x, 0)  # determine the size of the batch
            if current_batch_size == val_batch_size:
                val_loss_value, val_metric_value, val_sat_ratio = \
                    validation_check(model=model,
                                     inputs=x,
                                     targets=y,
                                     batch_size=current_batch_size,
                                     output_dim=output_dim,
                                     neur_act=neur_act,
                                     bound_sm_eq=ind_tensors_val[0],
                                     bound_sm_st=ind_tensors_val[1],
                                     bound_la_eq=ind_tensors_val[2],
                                     bound_la_st=ind_tensors_val[3],
                                     ind_sm_eq=ind_tensors_val[4],
                                     ind_sm_st=ind_tensors_val[5],
                                     ind_la_eq=ind_tensors_val[6],
                                     ind_la_st=ind_tensors_val[7],
                                     signed_indicator_food_expense=ind_tensors_val[8],
                                     signed_indicator_income=ind_tensors_val[9],
                                     lower_bound_output_variables=lower_bound_output_variables,
                                     upper_bound_output_variables=upper_bound_output_variables,
                                     lower_bound_input_variables=lower_bound_input_variables,
                                     upper_bound_input_variables=upper_bound_input_variables,
                                     undo_norm_direction=undo_norm_direction)

            else:
                val_loss_value, val_metric_value, val_sat_ratio = \
                    validation_check(model=model,
                                     inputs=x,
                                     targets=y,
                                     batch_size=current_batch_size,
                                     output_dim=output_dim,
                                     neur_act=neur_act,
                                     bound_sm_eq=ind_tensors_val_last[0],
                                     bound_sm_st=ind_tensors_val_last[1],
                                     bound_la_eq=ind_tensors_val_last[2],
                                     bound_la_st=ind_tensors_val_last[3],
                                     ind_sm_eq=ind_tensors_val_last[4],
                                     ind_sm_st=ind_tensors_val_last[5],
                                     ind_la_eq=ind_tensors_val_last[6],
                                     ind_la_st=ind_tensors_val_last[7],
                                     signed_indicator_food_expense=ind_tensors_val_last[8],
                                     signed_indicator_income=ind_tensors_val_last[9],
                                     lower_bound_output_variables=lower_bound_output_variables,
                                     upper_bound_output_variables=upper_bound_output_variables,
                                     lower_bound_input_variables=lower_bound_input_variables,
                                     upper_bound_input_variables=upper_bound_input_variables,
                                     undo_norm_direction=undo_norm_direction)

            # track progress from validation batch
            epoch_val_loss_average.update_state(values=val_loss_value, sample_weight=current_batch_size / train_batch_size)
            epoch_val_metric_average.update_state(values=val_metric_value, sample_weight=current_batch_size / train_batch_size)
            epoch_val_satisfaction_average.update_state(values=val_sat_ratio, sample_weight=current_batch_size / train_batch_size)


        # track progress from epoch
        val_loss_results.append(epoch_val_loss_average.result())
        val_metric_results.append(epoch_val_metric_average.result())
        val_satisfaction_results.append(epoch_val_satisfaction_average.result())

        print("Epoch {:03d}: Loss: {:.8f}, Mean absolute error: {:.8f}, Satisfaction rate val: {:.8%}, Satisfaction rate train: {:.8%}.".format(epoch,
                                                                                                          epoch_val_loss_average.result(),
                                                                                                          epoch_val_metric_average.result(),
                                                                                                          epoch_val_satisfaction_average.result(),
                                                                                                          epoch_train_satisfaction_average.result()))

        # save model if it satisfies sufficient amount of constraints and has a better loss (over the whole epoch) than the previous model
        if epoch_val_satisfaction_average.result() >= acceptable_sat_ratio and epoch_val_loss_average.result() < best_loss_average:
            best_loss_average = epoch_val_loss_average.result()
            best_metric_average = epoch_val_metric_average.result()
            print("Saving model.")
            model.save_weights(filepath=save_location_model+save_file_model)
            best_learning_rate = learning_rate_loss
            number_con = 0
            saved_model = True

        # process training stats and adjust parameters when necessary
        if epoch_val_satisfaction_average.result() >= acceptable_sat_ratio:     # check if satisfaction ratio is sufficiently high for
                                                                                # resulting in an acceptable model
            if first_time_loss == 1:  # check if previous step was for optimizing constraints
                # if so, adjust learning rate back to the one corresponding to loss function
                new_learning_rate = learning_rate_loss
                tf.compat.v1.variables_initializer(optimizer.variables())  # reset momentum of optimizer
                optimizer.lr.assign(learning_rate_loss)
                print('New learning rate: ' + str(learning_rate_loss) + '. Reason: learning rate loss needed.')
                changed_learning_rate = True
                first_time_loss += 1

            # check how close the two learning rates are and if previous step was for optimizing constraints
            if (learning_rate_loss - learning_rate_constraints)/learning_rate_loss < 0.1 and first_time_loss == 1:
                learning_rate_constraints = learning_rate_loss  # difference is small so reset lr constraints back to lr loss function
            elif first_time_loss == 1:
                # difference is large and reset lr constraints to median of lrs used in previous period for optimizing constraints
                learning_rate_constraints = learning_rate_constraints * (2**(np.floor(times_reduced_constraints/2)))

            # check if performance is constant on validation set
            if constant_performance_val % 15 == 0:
                # if so, decrease lr loss function
                learning_rate_loss = learning_rate_loss/2
                new_learning_rate = learning_rate_loss
                optimizer.lr.assign(learning_rate_loss)
                print('New learning rate: ' + str(learning_rate_loss) + '. Reason: constant performance on validation set.')
                changed_learning_rate = True

            # check if new model performed worse
            if epoch_val_loss_average.result() >= prev_val_loss_average:
                # if so, add counter for constant performance
                constant_performance_val += 1
            else:
                # if not, reset counter for constant performance
                constant_performance_val = 1

            # reset counters for optimizing constraints
            counter_optimize_constraints = 1
            times_reduced_constraints = 0
            first_time_con = 1

            acceptable_model = True

            # reset booleans for stopping and resetting criterion
            potential_bump_present = False

        else:
            # update epoch counter constraints
            epoch_constraints += 1
            # check if previous epoch optimized only the loss function
            if first_time_con == 1:
                # potential bump is possible
                potential_bump_present = True
                # update lr to lr of constraints
                if learning_rate_constraints < learning_rate_loss:
                    # if lr constraints is smaller, use this
                    new_learning_rate = learning_rate_constraints
                    tf.compat.v1.variables_initializer(optimizer.variables())  # reset momentum of optimizer
                    optimizer.lr.assign(learning_rate_constraints)
                    print('New learning rate: ' + str(learning_rate_constraints) + '. Reason: constrained learning rate needed.')
                    changed_learning_rate = True
                else:
                    # if lr loss is smaller, update lr constraints and use this updated lr
                    learning_rate_constraints = learning_rate_loss
                    new_learning_rate = learning_rate_constraints
                    tf.compat.v1.variables_initializer(optimizer.variables())  # reset momentum of optimizer
                    optimizer.lr.assign(learning_rate_constraints)
                    print('New learning rate: ' + str(learning_rate_constraints) + '. Reason: constrained learning rate needed.')
                    changed_learning_rate = True
                first_time_con += 1  # update parameter indicating that this step is constrained optimization
            else:
                # potential bump is not possible
                potential_bump_present = False
            if counter_optimize_constraints % 10 == 0:   # decrease lr after every 10 steps of constrained optimization (without any
                                                        # optimization of loss function in between)
                if times_reduced_constraints < 5:  # first 5 times reduce lr by halve
                    learning_rate_constraints = learning_rate_constraints/2
                    times_reduced_constraints += 1
                    new_learning_rate = learning_rate_constraints
                    tf.compat.v1.variables_initializer(optimizer.variables())  # reset momentum of optimizer
                    optimizer.lr.assign(learning_rate_constraints)
                    print('New learning rate: ' + str(learning_rate_constraints) + '. Reason: subtle optimization of constraints.')
                    changed_learning_rate = True
                else:  # the optimization is too slow, so make lr bigger again to speed up optimization of constraints
                    learning_rate_constraints = learning_rate_constraints * (2 ** times_reduced_constraints)
                    if learning_rate_constraints > learning_rate_loss:  # make sure the bound is respected
                        learning_rate_constraints = learning_rate_loss
                    times_reduced_constraints = 0
                    new_learning_rate = learning_rate_constraints
                    optimizer.lr.assign(learning_rate_constraints)
                    print('New learning rate: ' + str(learning_rate_constraints) + '. Reason: to slow subtle optimization of constraints.')
                    changed_learning_rate = True

            if epoch_val_satisfaction_average.result() > 0.75:  # only count these epochs where the satisfaction range is sufficiently high
                                                                # and reset if satisfaction is too low (for low satisfaction epochs one
                                                                # wants to use a large learning rate and does not want to think about subtle
                                                                # changes to the model)
                counter_optimize_constraints += 1
            else:
                counter_optimize_constraints = 1

            # reset counters for optimizing loss function
            constant_performance_val = 1
            first_time_loss = 1

            acceptable_model = False

        # compare parameters to check for a bump
        if epoch_val_loss_average.result() > prev_val_loss_average and potential_bump_present:
            bump_present = True
            epoch_bumps.append(epoch)
        else:
            bump_present = False

        # compare parameters to check constant performance
        if epoch_val_loss_average.result() >= prev_val_loss_average and acceptable_model:
            number_con += 1
            if number_bumps > 0:
                number_bump_con += 1

        # update the previous best parameters or overwrite when learning rate is changed
        if epoch_train_satisfaction_average.result() < prev_train_sat_average or changed_learning_rate:
            prev_train_sat_average = epoch_train_satisfaction_average.result()
        if epoch_train_loss_average.result() < prev_train_loss_average or changed_learning_rate:
            prev_train_loss_average = epoch_train_loss_average.result()
        if epoch_train_metric_average.result() < prev_train_met_average or changed_learning_rate:
            prev_train_met_average = epoch_train_metric_average.result()

        if epoch_val_satisfaction_average.result() < prev_val_sat_average or changed_learning_rate:
            prev_val_sat_average = epoch_val_satisfaction_average.result()
        if epoch_val_loss_average.result() < prev_val_loss_average or changed_learning_rate:
            prev_val_loss_average = epoch_val_loss_average.result()
        if epoch_val_metric_average.result() < prev_val_met_average or changed_learning_rate:
            prev_val_met_average = epoch_val_metric_average.result()

        list_lr_loss.append(learning_rate_loss)
        list_lr_con.append(learning_rate_constraints)
        list_lr.append(new_learning_rate)

        # check for stops and resets of algorithm
        reset_present = False  # revert the boolean to False before checking if it can be true
        if (number_bumps > max_bumps or number_bump_con > max_time_after_bump) and number_con <= max_constant_perf:
            reset_present = True
        if number_con > max_constant_perf or (number_resets > max_restarts and reset_present):
            if number_con > max_constant_perf:
                print('Perform early stopping due to constant performance.')  # print reason for early stopping
            else:
                print('Perform early stopping due to max number of resets.')  # print reason for early stopping
            model.load_weights(filepath=save_location_model+save_file_model)  # load weights of best model
            break  # perform early stopping

        # adjust counters
        if bump_present:
            number_bumps += 1
            number_bump_con = 0
            number_con = 0

        # adjust counters and load previous best model with new learning rate
        if reset_present:
            number_resets += 1
            number_bumps = 0
            number_bump_con = 0
            number_con = 0

            model.load_weights(save_location_model+save_file_model)
            new_learning_rate = best_learning_rate / 2
            optimizer.lr.assign(new_learning_rate)
            print('Reset to best model and new learning rate ' + str(new_learning_rate) + '.')
            epoch_resets.append(epoch)  # save epoch in which the reset was done.

        # reset metric states
        epoch_train_loss_average.reset_states()
        epoch_train_metric_average.reset_states()
        epoch_train_satisfaction_average.reset_states()

        epoch_val_loss_average.reset_states()
        epoch_val_metric_average.reset_states()
        epoch_val_satisfaction_average.reset_states()

        epoch += 1

        timing.append(time.time())

    if saved_model:
        model.load_weights(filepath=save_location_model + save_file_model)
    else:
        model.save_weights(filepath=save_location_model + save_file_model)

    if saved_model:
        print('----------------TRAINING-IS-DONE----------------------')
        print('Total number of epochs is ' + str(epoch) + '.')
        print('Constraints epoch is ' + str(epoch_constraints) + '.')
        print('Best validation loss is ' + str(best_loss_average.numpy()) + '.')
        print('Best metric loss is ' + str(best_metric_average.numpy()) + '.')
    else:
        print('----------------TRAINING-IS-DONE----------------------')
        print('Total number of epochs is ' + str(epoch) + '.')
        print('Constraints epoch is ' + str(epoch_constraints) + '.')
        print('Best validation loss is ' + str(val_loss_results[-1]) + '.')
        print('Best metric loss is ' + str(val_metric_results[-1]) + '.')

    return model, train_loss_results, train_metric_results, train_satisfaction_results, val_loss_results, val_metric_results, val_satisfaction_results, epoch, list_lr, list_lr_loss, list_lr_con, timing


def train_model_unconstrained(model, optimizer, train_number, train_dataset, train_batch_size, val_number, val_dataset, val_batch_size,
                              max_epochs, learning_rate, max_constant_perf, save_location_model, input_dim, output_dim, neur_act,
                              list_bound_constraints, save_file_model, lower_bound_output_variables, upper_bound_output_variables,
                              lower_bound_input_variables, upper_bound_input_variables):
    """Train model without constraints.

    One can observe that this function is not the fasted possible way to train a model without constraints. But this function should be
    used to compare the performance of the unconstrained model with the performance of the constrained model. Therefore it is interesting
    to also take into account the satisfaction ratio of the constraints. Hence the only sub-optimal thing in this function is the
    computation of the gradients of these constraints. But as long as this function is not used for comparing time-efficiency this is
    not a problem. If one wants to compare the time efficiency together with the satisfaction ratio, then one should make a new function
    that does the same as compute_sum_exp1_exp2_grad, but the gradient tape should not be persistent and only the first tape.gradients
    should be maintained. All the other gradient computations should be removed. In the case that one wants to check the satisfaction ratio
    only after training, then one can use a standard GradientTape of the loss function to compute the gradients.

    Parameters
    ----------
    model : tf.keras.Model
        Initial model to start the training from.
    optimizer : tf.keras.Optimizers
        Optimizer used to updated each iteration.
    train_number : int
        Integer denoting the total number of training examples.
    train_dataset : tf.data.Dataset
        Dataset containing the input and ground truths of the examples in the training set.
    train_batch_size : int
        Integer denoting the batch_size of the training set.
    val_number : int
        Integer denoting the total number of validation examples.
    val_dataset : tf.data.Dataset
        Dataset containing the input and ground truths of the examples in the validation set.
    val_batch_size : int
        Integer denoting the batch_size of the validation set.
    max_epochs : int
        Integer denoting the maximum number of epochs to train the model.
    learning_rate : float
        Initial value of the learning rate used by the optimizer.
    max_constant_perf : int
        Integer denoting the maximum number of epochs where the model does not improve and is allowed to train further. This value is used
        for early stopping.
    save_location_model : string
        Directory for saving the model weights.
    input_dim : int
        Integer denoting the input dimension of the model.
    output_dim : int
        Integer denoting the output dimension of the model.
    neur_act : list[int]
        List containing the dimensions of the weight matrices and the number of acitvation functions for each layer.
    list_bound_constraints : list[list[tf.Tensor]]
        List containing the bound constraints with the specified syntax.
    save_file_model : string
        Name of the file containing the weights of the trained model.
    lower_bound_output_variables : tf.Tensor
        Tensor of shape=(1, output_dimension) and dtype=tf.float32 denoting the lower bound used for the normalization of each output
        variable.
    upper_bound_output_variables : tf.Tensor
        Tensor of shape=(1, output_dimension) and dtype=tf.float32 denoting the upper bound used for the normalization of each output
        variable.
    lower_bound_input_variables : tf.Tensor
        Tensor of shape=(1, input_dimension) and dtype=tf.float32 denoting the lower bound used for the normalization of each input
        variable.
    upper_bound_input_variables : tf.Tensor
        Tensor of shape=(1, input_dimension) and dtype=tf.float32 denoting the upper bound used for the normalization of each input
        variable.

    Returns
    ----------
    model : tf.keras.Model
        Model obtained from training.
    train_loss_results : list[tf.Tensor]
        List containing the value of the loss function of the training set for each epoch.
    train_metric_results : list[tf.Tensor]
        List containing the value of the metric of the training set for each epoch.
    train_satisfaction_results : list[tf.Tensor]
        List containing the satisfaction ratio of the training set for each epoch.
    val_loss_results : list[tf.Tensor]
        List containing the value of the loss function of the validation set for each epoch.
    val_metric_results : list[tf.Tensor]
        List containing the value of the metric of the validation set for each epoch.
    val_satisfaction_results : list[tf.Tensor]
        List containing the satisfaction ratio of the validation set for each epoch.
    epoch : int
        The number of epochs trained for
    list_lr_loss : list[float]
        List containing the learning rate for each epoch.
    timing : list[float]
        List containing the end time of each epoch.

    """
    ### check for existence of directory for saving model weights
    if not os.path.isdir(save_location_model):
        os.makedirs(save_location_model)
    ### initialize constant tensors for computing satisfaction ratio of constraints
    # tensors for train set
    ind_tensors_train = initialize_indicator_tensors(batch_size=train_batch_size,
                                                     output_dim=output_dim,
                                                     neur_act=neur_act,
                                                     list_bound_constraints=list_bound_constraints)

    if not train_number % train_batch_size == 0:  # for the final batch that is smaller than the others, the constant tensors containing
        # bounds and indicators need to be smaller
        new_train_batch_size = train_number % train_batch_size
        ind_tensors_train_last = initialize_indicator_tensors(batch_size=new_train_batch_size,
                                                              output_dim=output_dim,
                                                              neur_act=neur_act,
                                                              list_bound_constraints=list_bound_constraints)

    # tensors for validation set (only when the user gave a validation set)
    ind_tensors_val = initialize_indicator_tensors(batch_size=val_batch_size,
                                                   output_dim=output_dim,
                                                   neur_act=neur_act,
                                                   list_bound_constraints=list_bound_constraints)

    if not val_number % val_batch_size == 0:
        new_val_batch_size = val_number % val_batch_size
        ind_tensors_val_last = initialize_indicator_tensors(batch_size=new_val_batch_size,
                                                            output_dim=output_dim,
                                                            neur_act=neur_act,
                                                            list_bound_constraints=list_bound_constraints)

    lower_bound_output_variables = tf.reshape(tf.convert_to_tensor(lower_bound_output_variables), [1, output_dim])
    upper_bound_output_variables = tf.reshape(tf.convert_to_tensor(upper_bound_output_variables), [1, output_dim])
    lower_bound_input_variables = tf.reshape(tf.convert_to_tensor(lower_bound_input_variables), [1, 1])
    upper_bound_input_variables = tf.reshape(tf.convert_to_tensor(upper_bound_input_variables), [1, 1])
    undo_norm_direction = tf.math.subtract(upper_bound_output_variables, lower_bound_output_variables)

    ### define variables that are used throughout the training procedure
    # initialize lists for recording results
    train_loss_results = []
    train_metric_results = []
    train_satisfaction_results = []

    val_loss_results = []
    val_metric_results = []
    val_satisfaction_results = []

    # epoch numbers
    epoch = 1

    # define learning_rates
    learning_rate_loss = learning_rate
    list_lr_loss = [learning_rate_loss]

    # initialize metrics for recording values of loss function, additional metrics and satisfaction ratio for each batch in an epoch
    epoch_train_loss_average = tf.keras.metrics.Mean()
    epoch_train_metric_average = tf.keras.metrics.Mean()
    epoch_train_satisfaction_average = tf.keras.metrics.Mean()

    epoch_val_loss_average = tf.keras.metrics.Mean()
    epoch_val_metric_average = tf.keras.metrics.Mean()
    epoch_val_satisfaction_average = tf.keras.metrics.Mean()

    # initialize previous values of loss function, additional metrics and satisfaction ratio
    prev_train_sat_average = 10 ** 3
    prev_train_loss_average = 10 ** 3
    prev_train_met_average = 10 ** 3

    prev_val_sat_average = 10 ** 3
    prev_val_loss_average = 10 ** 3
    prev_val_met_average = 10 ** 3

    best_loss_average = 10 ** 3
    best_metric_average = 10 ** 3

    constant_loss = 1

    timing = []  # initialize list of times of the epochs of the training process

    timing.append(time.time())

    ### start of training procedure
    while epoch <= max_epochs:
        for x, y in train_dataset:
            current_batch_size = np.size(x, 0)  # determine the size of the batch
            if current_batch_size == train_batch_size:
                # optimize model
                gradient_loss, gradient_smaller_equal, gradient_smaller_strict, gradient_larger_equal, gradient_larger_strict, gradient_sum_constraint_food_expense, gradient_sum_constraint_income, train_loss_value, train_metric_value, train_satisfaction_ratio \
                    = compute_sum_grad_undo_norm(model=model,
                                                 inputs=x,
                                                 targets=y,
                                                 batch_size=current_batch_size,
                                                 output_dim=output_dim,
                                                 neur_act=neur_act,
                                                 bound_sm_eq=ind_tensors_train[0],
                                                 bound_sm_st=ind_tensors_train[1],
                                                 bound_la_eq=ind_tensors_train[2],
                                                 bound_la_st=ind_tensors_train[3],
                                                 ind_sm_eq=ind_tensors_train[4],
                                                 ind_sm_st=ind_tensors_train[5],
                                                 ind_la_eq=ind_tensors_train[6],
                                                 ind_la_st=ind_tensors_train[7],
                                                 signed_indicator_food_expense=ind_tensors_train[8],
                                                 signed_indicator_income=ind_tensors_train[9],
                                                 lower_bound_output_variables=lower_bound_output_variables,
                                                 upper_bound_output_variables=upper_bound_output_variables,
                                                 lower_bound_input_variables=lower_bound_input_variables,
                                                 upper_bound_input_variables=upper_bound_input_variables,
                                                 undo_norm_direction=undo_norm_direction)

                optimizer.apply_gradients(zip(gradient_loss, model.trainable_variables))
            else:  # use the constant tensors and indicator tensors with size corresponding to the last (and smaller) batch
                # optimize model
                gradient_loss, gradient_smaller_equal, gradient_smaller_strict, gradient_larger_equal, gradient_larger_strict, gradient_sum_constraint_food_expense, gradient_sum_constraint_income, train_loss_value, train_metric_value, train_satisfaction_ratio \
                    = compute_sum_grad_undo_norm(model=model,
                                                 inputs=x,
                                                 targets=y,
                                                 batch_size=current_batch_size,
                                                 output_dim=output_dim,
                                                 neur_act=neur_act,
                                                 bound_sm_eq=ind_tensors_train_last[0],
                                                 bound_sm_st=ind_tensors_train_last[1],
                                                 bound_la_eq=ind_tensors_train_last[2],
                                                 bound_la_st=ind_tensors_train_last[3],
                                                 ind_sm_eq=ind_tensors_train_last[4],
                                                 ind_sm_st=ind_tensors_train_last[5],
                                                 ind_la_eq=ind_tensors_train_last[6],
                                                 ind_la_st=ind_tensors_train_last[7],
                                                 signed_indicator_food_expense=ind_tensors_train_last[8],
                                                 signed_indicator_income=ind_tensors_train_last[9],
                                                 lower_bound_output_variables=lower_bound_output_variables,
                                                 upper_bound_output_variables=upper_bound_output_variables,
                                                 lower_bound_input_variables=lower_bound_input_variables,
                                                 upper_bound_input_variables=upper_bound_input_variables,
                                                 undo_norm_direction=undo_norm_direction)
                optimizer.apply_gradients(zip(gradient_loss, model.trainable_variables))

            # track progress from training batch
            epoch_train_loss_average.update_state(values=train_loss_value, sample_weight=current_batch_size/train_batch_size)
            epoch_train_metric_average.update_state(values=train_metric_value, sample_weight=current_batch_size/train_batch_size)
            epoch_train_satisfaction_average.update_state(values=train_satisfaction_ratio, sample_weight=current_batch_size/train_batch_size)

        # track progress from epoch
        train_loss_results.append(epoch_train_loss_average.result())
        train_metric_results.append(epoch_train_metric_average.result())
        train_satisfaction_results.append(epoch_train_satisfaction_average.result())

        # check performance on validation set
        for x, y in val_dataset:
            current_batch_size = np.size(x, 0)  # determine the size of the batch
            if current_batch_size == val_batch_size:
                val_loss_value, val_metric_value, val_sat_ratio = \
                    validation_check(model=model,
                                     inputs=x,
                                     targets=y,
                                     batch_size=current_batch_size,
                                     output_dim=output_dim,
                                     neur_act=neur_act,
                                     bound_sm_eq=ind_tensors_val[0],
                                     bound_sm_st=ind_tensors_val[1],
                                     bound_la_eq=ind_tensors_val[2],
                                     bound_la_st=ind_tensors_val[3],
                                     ind_sm_eq=ind_tensors_val[4],
                                     ind_sm_st=ind_tensors_val[5],
                                     ind_la_eq=ind_tensors_val[6],
                                     ind_la_st=ind_tensors_val[7],
                                     signed_indicator_food_expense=ind_tensors_val[8],
                                     signed_indicator_income=ind_tensors_val[9],
                                     lower_bound_output_variables=lower_bound_output_variables,
                                     upper_bound_output_variables=upper_bound_output_variables,
                                     lower_bound_input_variables=lower_bound_input_variables,
                                     upper_bound_input_variables=upper_bound_input_variables,
                                     undo_norm_direction=undo_norm_direction)

            else:
                val_loss_value, val_metric_value, val_sat_ratio = \
                    validation_check(model=model,
                                     inputs=x,
                                     targets=y,
                                     batch_size=current_batch_size,
                                     output_dim=output_dim,
                                     neur_act=neur_act,
                                     bound_sm_eq=ind_tensors_val_last[0],
                                     bound_sm_st=ind_tensors_val_last[1],
                                     bound_la_eq=ind_tensors_val_last[2],
                                     bound_la_st=ind_tensors_val_last[3],
                                     ind_sm_eq=ind_tensors_val_last[4],
                                     ind_sm_st=ind_tensors_val_last[5],
                                     ind_la_eq=ind_tensors_val_last[6],
                                     ind_la_st=ind_tensors_val_last[7],
                                     signed_indicator_food_expense=ind_tensors_val_last[8],
                                     signed_indicator_income=ind_tensors_val_last[9],
                                     lower_bound_output_variables=lower_bound_output_variables,
                                     upper_bound_output_variables=upper_bound_output_variables,
                                     lower_bound_input_variables=lower_bound_input_variables,
                                     upper_bound_input_variables=upper_bound_input_variables,
                                     undo_norm_direction=undo_norm_direction
                                     )

            # track progress from validation batch
            epoch_val_loss_average.update_state(values=val_loss_value, sample_weight=current_batch_size/val_batch_size)
            epoch_val_metric_average.update_state(values=val_metric_value, sample_weight=current_batch_size/val_batch_size)
            epoch_val_satisfaction_average.update_state(values=val_sat_ratio, sample_weight=current_batch_size/val_batch_size)

        # track progress from epoch
        val_loss_results.append(epoch_val_loss_average.result())
        val_metric_results.append(epoch_val_metric_average.result())
        val_satisfaction_results.append(epoch_val_satisfaction_average.result())

        print("Epoch {:03d}: Loss: {:.8f}, Mean absolute error: {:.8f}, Satisfaction rate val: {:.8%}, Satisfaction rate train: {:.8%}.".format(epoch,
                                                                                                            epoch_val_loss_average.result(),
                                                                                                            epoch_val_metric_average.result(),
                                                                                                            epoch_val_satisfaction_average.result(),
                                                                                                            epoch_train_satisfaction_average.result()))

        # save model if its performance is better
        if epoch_val_loss_average.result() < best_loss_average:
            best_loss_average = epoch_val_loss_average.result()
            best_metric_average = epoch_val_metric_average.result()
            print("Saving model.")
            model.save_weights(filepath=save_location_model+save_file_model)
            best_learning_rate = learning_rate_loss

        if epoch_val_loss_average.result() > prev_val_loss_average:
            constant_loss += 1
        else:
            constant_loss = 1

        if constant_loss % 15 == 0:
            learning_rate_loss = learning_rate_loss / 2
            print('New learning rate: ' + str(learning_rate_loss) + '. Reason: constant performance on validation set.')
            optimizer.lr.assign(learning_rate_loss)

        if constant_loss == max_constant_perf:
            print('Too long constant performance on validation set. Perform early stopping.')
            model.load_weights(filepath=save_location_model+save_file_model)
            break

        # update the previous best parameters or overwrite when learning rate is changed
        if epoch_train_satisfaction_average.result() < prev_train_sat_average:
            prev_train_sat_average = epoch_train_satisfaction_average.result()
        if epoch_train_loss_average.result() < prev_train_loss_average:
            prev_train_loss_average = epoch_train_loss_average.result()
        if epoch_train_metric_average.result() < prev_train_met_average:
            prev_train_met_average = epoch_train_metric_average.result()

        if epoch_val_satisfaction_average.result() < prev_val_sat_average:
            prev_val_sat_average = epoch_val_satisfaction_average.result()
        if epoch_val_loss_average.result() < prev_val_loss_average:
            prev_val_loss_average = epoch_val_loss_average.result()
        if epoch_val_metric_average.result() < prev_val_met_average:
            prev_val_met_average = epoch_val_metric_average.result()

        list_lr_loss.append(learning_rate_loss)

        # reset metric states
        epoch_train_loss_average.reset_states()
        epoch_train_metric_average.reset_states()
        epoch_train_satisfaction_average.reset_states()

        epoch_val_loss_average.reset_states()
        epoch_val_metric_average.reset_states()
        epoch_val_satisfaction_average.reset_states()

        epoch += 1

        timing.append(time.time())

    model.load_weights(filepath=save_location_model + save_file_model)

    print('----------------TRAINING-IS-DONE----------------------')
    print('Total number of epochs is ' + str(epoch) + '.')
    print('Best validation loss is ' + str(best_loss_average.numpy()) + '.')
    print('Best metric loss is ' + str(best_metric_average.numpy()) + '.')

    return model, train_loss_results, train_metric_results, train_satisfaction_results, val_loss_results, val_metric_results, val_satisfaction_results, epoch, list_lr_loss, timing


def train_model_unconstrained_only_label(model, optimizer, train_number, train_dataset, train_batch_size, val_number, val_dataset,
                                         val_batch_size, max_epochs, learning_rate, max_constant_perf, save_location_model, input_dim,
                                         output_dim, neur_act, list_bound_constraints, sup_batches, save_file_model,
                                         lower_bound_output_variables, upper_bound_output_variables, lower_bound_input_variables,
                                         upper_bound_input_variables):
    """Train model without constraints.

    One can observe that this function is not the fasted possible way to train a model without constraints. But this function should be
    used to compare the performance of the unconstrained model with the performance of the constrained model. Therefore it is interesting
    to also take into account the satisfaction ratio of the constraints. Hence the only sub-optimal thing in this function is the
    computation of the gradients of these constraints. But as long as this function is not used for comparing time-efficiency this is
    not a problem. If one wants to compare the time efficiency together with the satisfaction ratio, then one should make a new function
    that does the same as compute_sum_exp1_exp2_grad, but the gradient tape should not be persistent and only the first tape.gradients
    should be maintained. All the other gradient computations should be removed. In the case that one wants to check the satisfaction ratio
    only after training, then one can use a standard GradientTape of the loss function to compute the gradients.

    Parameters
    ----------
    model : tf.keras.Model
        Initial model to start the training from.
    optimizer : tf.keras.Optimizers
        Optimizer used to updated each iteration.
    train_number : int
        Integer denoting the total number of training examples.
    train_dataset : tf.data.Dataset
        Dataset containing the input and ground truths of the examples in the training set.
    train_batch_size : int
        Integer denoting the batch_size of the training set.
    val_number : int
        Integer denoting the total number of validation examples.
    val_dataset : tf.data.Dataset
        Dataset containing the input and ground truths of the examples in the validation set.
    val_batch_size : int
        Integer denoting the batch_size of the validation set.
    max_epochs : int
        Integer denoting the maximum number of epochs to train the model.
    learning_rate : float
        Initial value of the learning rate used by the optimizer.
    max_constant_perf : int
        Integer denoting the maximum number of epochs where the model does not improve and is allowed to train further. This value is used
        for early stopping.
    save_location_model : string
        Directory for saving the model weights.
    input_dim : int
        Integer denoting the input dimension of the model.
    output_dim : int
        Integer denoting the output dimension of the model.
    neur_act : list[int]
        List containing the dimensions of the weight matrices and the number of acitvation functions for each layer.
    list_bound_constraints : list[list[tf.Tensor]]
        List containing the bound constraints with the specified syntax.
    sup_batches : list[int]
        List containing integers denoting the indices of which batches are supervised. The other batches are trained unsupervised.
    save_file_model : string
        Name of file containing the trained model weights.
    lower_bound_output_variables : tf.Tensor
        Tensor of shape=(1, output_dimension) and dtype=tf.float32 denoting the lower bound used for the normalization of each output
        variable.
    upper_bound_output_variables : tf.Tensor
        Tensor of shape=(1, output_dimension) and dtype=tf.float32 denoting the upper bound used for the normalization of each output
        variable.
    lower_bound_input_variables : tf.Tensor
        Tensor of shape=(1, input_dimension) and dtype=tf.float32 denoting the lower bound used for the normalization of each input
        variable.
    upper_bound_input_variables : tf.Tensor
        Tensor of shape=(1, input_dimension) and dtype=tf.float32 denoting the upper bound used for the normalization of each input
        variable.

    Returns
    ----------
    model : tf.keras.Model
        Model obtained from training.
    train_loss_results : list[tf.Tensor]
        List containing the value of the loss function of the training set for each epoch.
    train_metric_results : list[tf.Tensor]
        List containing the value of the metric of the training set for each epoch.
    train_satisfaction_results : list[tf.Tensor]
        List containing the satisfaction ratio of the training set for each epoch.
    val_loss_results : list[tf.Tensor]
        List containing the value of the loss function of the validation set for each epoch.
    val_metric_results : list[tf.Tensor]
        List containing the value of the metric of the validation set for each epoch.
    val_satisfaction_results : list[tf.Tensor]
        List containing the satisfaction ratio of the validation set for each epoch.
    epoch : int
        The number of epochs trained for
    list_lr_loss : list[float]
        List containing the learning rate for each epoch.
    timing : list[float]
        List containing the end time of each epoch.

    """
    ### check for existence of directory for saving model weights
    if not os.path.isdir(save_location_model):
        os.makedirs(save_location_model)
    ### initialize constant tensors for computing satisfaction ratio of constraints
    # tensors for train set
    ind_tensors_train = initialize_indicator_tensors(batch_size=train_batch_size,
                                                     output_dim=output_dim,
                                                     neur_act=neur_act,
                                                     list_bound_constraints=list_bound_constraints)

    if not train_number % train_batch_size == 0:  # for the final batch that is smaller than the others, the constant tensors containing
        # bounds and indicators need to be smaller
        new_train_batch_size = train_number % train_batch_size
        ind_tensors_train_last = initialize_indicator_tensors(batch_size=new_train_batch_size,
                                                              output_dim=output_dim,
                                                              neur_act=neur_act,
                                                              list_bound_constraints=list_bound_constraints)

    # tensors for validation set (only when the user gave a validation set)
    ind_tensors_val = initialize_indicator_tensors(batch_size=val_batch_size,
                                                   output_dim=output_dim,
                                                   neur_act=neur_act,
                                                   list_bound_constraints=list_bound_constraints)

    if not val_number % val_batch_size == 0:
        new_val_batch_size = val_number % val_batch_size
        ind_tensors_val_last = initialize_indicator_tensors(batch_size=new_val_batch_size,
                                                            output_dim=output_dim,
                                                            neur_act=neur_act,
                                                            list_bound_constraints=list_bound_constraints)

    lower_bound_output_variables = tf.reshape(tf.convert_to_tensor(lower_bound_output_variables), [1, output_dim])
    upper_bound_output_variables = tf.reshape(tf.convert_to_tensor(upper_bound_output_variables), [1, output_dim])
    lower_bound_input_variables = tf.reshape(tf.convert_to_tensor(lower_bound_input_variables), [1, 1])
    upper_bound_input_variables = tf.reshape(tf.convert_to_tensor(upper_bound_input_variables), [1, 1])
    undo_norm_direction = tf.math.subtract(upper_bound_output_variables, lower_bound_output_variables)

    ### define variables that are used throughout the training procedure
    # initialize lists for recording results
    train_loss_results = []
    train_metric_results = []
    train_satisfaction_results = []

    val_loss_results = []
    val_metric_results = []
    val_satisfaction_results = []

    # epoch numbers
    epoch = 1

    # define learning_rates
    learning_rate_loss = learning_rate
    list_lr_loss = [learning_rate_loss]

    # initialize metrics for recording values of loss function, additional metrics and satisfaction ratio for each batch in an epoch
    epoch_train_loss_average = tf.keras.metrics.Mean()
    epoch_train_metric_average = tf.keras.metrics.Mean()
    epoch_train_satisfaction_average = tf.keras.metrics.Mean()

    epoch_val_loss_average = tf.keras.metrics.Mean()
    epoch_val_metric_average = tf.keras.metrics.Mean()
    epoch_val_satisfaction_average = tf.keras.metrics.Mean()

    # initialize previous values of loss function, additional metrics and satisfaction ratio
    prev_train_sat_average = 10 ** 3
    prev_train_loss_average = 10 ** 3
    prev_train_met_average = 10 ** 3

    prev_val_sat_average = 10 ** 3
    prev_val_loss_average = 10 ** 3
    prev_val_met_average = 10 ** 3

    best_loss_average = 10 ** 3
    best_metric_average = 10 ** 3

    constant_loss = 1

    timing = []  # initialize list of times of the epochs of the training process

    timing.append(time.time())

    ### start of training procedure
    while epoch < max_epochs:
        counter_batches = 0
        for x, y in train_dataset:
            if counter_batches in sup_batches:
                current_batch_size = np.size(x, 0)  # determine the size of the batch
                if current_batch_size == train_batch_size:
                    # optimize model
                    gradient_loss, gradient_smaller_equal, gradient_smaller_strict, gradient_larger_equal, gradient_larger_strict, gradient_sum_constraint_food_expense, gradient_sum_constraint_income, train_loss_value, train_metric_value, train_satisfaction_ratio \
                        = compute_sum_grad_undo_norm(model=model,
                                                     inputs=x,
                                                     targets=y,
                                                     batch_size=current_batch_size,
                                                     output_dim=output_dim,
                                                     neur_act=neur_act,
                                                     bound_sm_eq=ind_tensors_train[0],
                                                     bound_sm_st=ind_tensors_train[1],
                                                     bound_la_eq=ind_tensors_train[2],
                                                     bound_la_st=ind_tensors_train[3],
                                                     ind_sm_eq=ind_tensors_train[4],
                                                     ind_sm_st=ind_tensors_train[5],
                                                     ind_la_eq=ind_tensors_train[6],
                                                     ind_la_st=ind_tensors_train[7],
                                                     signed_indicator_food_expense=ind_tensors_train[8],
                                                     signed_indicator_income=ind_tensors_train[9],
                                                     lower_bound_output_variables=lower_bound_output_variables,
                                                     upper_bound_output_variables=upper_bound_output_variables,
                                                     lower_bound_input_variables=lower_bound_input_variables,
                                                     upper_bound_input_variables=upper_bound_input_variables,
                                                     undo_norm_direction=undo_norm_direction)

                    optimizer.apply_gradients(zip(gradient_loss, model.trainable_variables))
                else:  # use the constant tensors and indicator tensors with size corresponding to the last (and smaller) batch
                    # optimize model
                    gradient_loss, gradient_smaller_equal, gradient_smaller_strict, gradient_larger_equal, gradient_larger_strict, gradient_sum_constraint_food_expense, gradient_sum_constraint_income, train_loss_value, train_metric_value, train_satisfaction_ratio \
                        = compute_sum_grad_undo_norm(model=model,
                                                     inputs=x,
                                                     targets=y,
                                                     batch_size=current_batch_size,
                                                     output_dim=output_dim,
                                                     neur_act=neur_act,
                                                     bound_sm_eq=ind_tensors_train_last[0],
                                                     bound_sm_st=ind_tensors_train_last[1],
                                                     bound_la_eq=ind_tensors_train_last[2],
                                                     bound_la_st=ind_tensors_train_last[3],
                                                     ind_sm_eq=ind_tensors_train_last[4],
                                                     ind_sm_st=ind_tensors_train_last[5],
                                                     ind_la_eq=ind_tensors_train_last[6],
                                                     ind_la_st=ind_tensors_train_last[7],
                                                     signed_indicator_food_expense=ind_tensors_train_last[8],
                                                     signed_indicator_income=ind_tensors_train_last[9],
                                                     lower_bound_output_variables=lower_bound_output_variables,
                                                     upper_bound_output_variables=upper_bound_output_variables,
                                                     lower_bound_input_variables=lower_bound_input_variables,
                                                     upper_bound_input_variables=upper_bound_input_variables,
                                                     undo_norm_direction=undo_norm_direction)
                    optimizer.apply_gradients(zip(gradient_loss, model.trainable_variables))

                # track progress from training batch
                epoch_train_loss_average.update_state(values=train_loss_value, sample_weight=current_batch_size/train_batch_size)
                epoch_train_metric_average.update_state(values=train_metric_value, sample_weight=current_batch_size/train_batch_size)
                epoch_train_satisfaction_average.update_state(values=train_satisfaction_ratio, sample_weight=current_batch_size/train_batch_size)

            counter_batches += 1

        # track progress from epoch
        train_loss_results.append(epoch_train_loss_average.result())
        train_metric_results.append(epoch_train_metric_average.result())
        train_satisfaction_results.append(epoch_train_satisfaction_average.result())

        # check performance on validation set
        for x, y in val_dataset:
            current_batch_size = np.size(x, 0)  # determine the size of the batch
            if current_batch_size == val_batch_size:
                val_loss_value, val_metric_value, val_sat_ratio = \
                    validation_check(model=model,
                                     inputs=x,
                                     targets=y,
                                     batch_size=current_batch_size,
                                     output_dim=output_dim,
                                     neur_act=neur_act,
                                     bound_sm_eq=ind_tensors_val[0],
                                     bound_sm_st=ind_tensors_val[1],
                                     bound_la_eq=ind_tensors_val[2],
                                     bound_la_st=ind_tensors_val[3],
                                     ind_sm_eq=ind_tensors_val[4],
                                     ind_sm_st=ind_tensors_val[5],
                                     ind_la_eq=ind_tensors_val[6],
                                     ind_la_st=ind_tensors_val[7],
                                     signed_indicator_food_expense=ind_tensors_val[8],
                                     signed_indicator_income=ind_tensors_val[9],
                                     lower_bound_output_variables=lower_bound_output_variables,
                                     upper_bound_output_variables=upper_bound_output_variables,
                                     lower_bound_input_variables=lower_bound_input_variables,
                                     upper_bound_input_variables=upper_bound_input_variables,
                                     undo_norm_direction=undo_norm_direction
                                     )

            else:
                val_loss_value, val_metric_value, val_sat_ratio = \
                    validation_check(model=model,
                                     inputs=x,
                                     targets=y,
                                     batch_size=current_batch_size,
                                     output_dim=output_dim,
                                     neur_act=neur_act,
                                     bound_sm_eq=ind_tensors_val_last[0],
                                     bound_sm_st=ind_tensors_val_last[1],
                                     bound_la_eq=ind_tensors_val_last[2],
                                     bound_la_st=ind_tensors_val_last[3],
                                     ind_sm_eq=ind_tensors_val_last[4],
                                     ind_sm_st=ind_tensors_val_last[5],
                                     ind_la_eq=ind_tensors_val_last[6],
                                     ind_la_st=ind_tensors_val_last[7],
                                     signed_indicator_food_expense=ind_tensors_val_last[8],
                                     signed_indicator_income=ind_tensors_val_last[9],
                                     lower_bound_output_variables=lower_bound_output_variables,
                                     upper_bound_output_variables=upper_bound_output_variables,
                                     lower_bound_input_variables=lower_bound_input_variables,
                                     upper_bound_input_variables=upper_bound_input_variables,
                                     undo_norm_direction=undo_norm_direction)

            # track progress from validation batch
            epoch_val_loss_average.update_state(values=val_loss_value, sample_weight=current_batch_size/val_batch_size)
            epoch_val_metric_average.update_state(values=val_metric_value, sample_weight=current_batch_size/val_batch_size)
            epoch_val_satisfaction_average.update_state(values=val_sat_ratio, sample_weight=current_batch_size/val_batch_size)
 
        # track progress from epoch
        val_loss_results.append(epoch_val_loss_average.result())
        val_metric_results.append(epoch_val_metric_average.result())
        val_satisfaction_results.append(epoch_val_satisfaction_average.result())

        print("Epoch {:03d}: Loss: {:.8f}, Mean absolute error: {:.8f}, Satisfaction rate val: {:.8%}, Satisfaction rate train: {:.8%}.".format(epoch,
                                                                                                            epoch_val_loss_average.result(),
                                                                                                            epoch_val_metric_average.result(),
                                                                                                            epoch_val_satisfaction_average.result(),
                                                                                                            epoch_train_satisfaction_average.result()))

        # save model if its performance is better
        if epoch_val_loss_average.result() < best_loss_average:
            best_loss_average = epoch_val_loss_average.result()
            best_metric_average = epoch_val_metric_average.result()
            print("Saving model.")
            model.save_weights(filepath=save_location_model+save_file_model)
            best_learning_rate = learning_rate_loss

        if epoch_val_loss_average.result() > prev_val_loss_average:
            constant_loss += 1
        else:
            constant_loss = 1

        if constant_loss % 15 == 0:
            learning_rate_loss = learning_rate_loss / 2
            print('New learning rate: ' + str(learning_rate_loss) + '. Reason: constant performance on validation set.')
            optimizer.lr.assign(learning_rate_loss)

        if constant_loss == max_constant_perf:
            print('Too long constant performance on validation set. Perform early stopping.')
            model.load_weights(filepath=save_location_model+save_file_model)
            break

        # update the previous best parameters or overwrite when learning rate is changed
        if epoch_train_satisfaction_average.result() < prev_train_sat_average:
            prev_train_sat_average = epoch_train_satisfaction_average.result()
        if epoch_train_loss_average.result() < prev_train_loss_average:
            prev_train_loss_average = epoch_train_loss_average.result()
        if epoch_train_metric_average.result() < prev_train_met_average:
            prev_train_met_average = epoch_train_metric_average.result()

        if epoch_val_satisfaction_average.result() < prev_val_sat_average:
            prev_val_sat_average = epoch_val_satisfaction_average.result()
        if epoch_val_loss_average.result() < prev_val_loss_average:
            prev_val_loss_average = epoch_val_loss_average.result()
        if epoch_val_metric_average.result() < prev_val_met_average:
            prev_val_met_average = epoch_val_metric_average.result()

        list_lr_loss.append(learning_rate_loss)

        # reset metric states
        epoch_train_loss_average.reset_states()
        epoch_train_metric_average.reset_states()
        epoch_train_satisfaction_average.reset_states()

        epoch_val_loss_average.reset_states()
        epoch_val_metric_average.reset_states()
        epoch_val_satisfaction_average.reset_states()

        epoch += 1

        timing.append(time.time())

    print('----------------TRAINING-IS-DONE----------------------')
    print('Total number of epochs is ' + str(epoch) + '.')
    print('Best validation loss is ' + str(best_loss_average.numpy()) + '.')
    print('Best metric loss is ' + str(best_metric_average.numpy()) + '.')

    return model, train_loss_results, train_metric_results, train_satisfaction_results, val_loss_results, val_metric_results, val_satisfaction_results, epoch, list_lr_loss, timing


def train_model_fuzzy(model, input_dim, output_dim, optimizer, neur_act, list_bound_constraints, train_number, train_dataset, train_batch_size,
                      val_number, val_dataset, val_batch_size, max_epochs, max_con_epochs, rescale_constant, learning_rate, acceptable_sat_ratio,
                      max_bumps, max_time_after_bump, max_constant_perf, max_restarts, save_location_model, save_file_model,
                      lower_bound_output_variables, upper_bound_output_variables, lower_bound_input_variables,
                      upper_bound_input_variables):
    """
    Parameters
    ----------
    model : tf.keras.Model
        model to be trained.
    input_dim : int
        integer denoting the input dimension of the network.
    output_dim : int
        integer denoting in the output dimension of the network.
    optimizer : tf.keras.Optimizer
        optimizer used during training.
    neur_act : list[int]
        list containing the first dimension of the weight matrices and the number of activation functions for each hidden layer.
    list_bound_constraints : list[list[tf.Tensor]]
        list containing the bound cosntraints.
    train_number : int
        integer denoting the number of training examples.
    train_dataset : tf.data.Dataset
        dataset containing the training examples.
    train_batch_size : int
        batch-size used for the training set.
    val_number : int
        integer denoting the number of validation examples.
    val_dataset : tf.data.Dataset
        dataset containing the validation examples.
    val_batch_size : int
        batch-size used for the validation set.
    max_epochs : int
        integer denoting the maximum number of epochs that can be used to optimize the loss function alone.
    max_con_epochs : int
        integer denoting the maximum number of epochs that can be used to optimize the constraints.
    rescale_constant : float
        factor to rescale the normalized gradients before multiplying with sup norm of gradients of loss function
    learning_rate : float
        float denoting the initial value of the learning rate of the optimizer.
    acceptable_sat_ratio : float
        float in [0,1] indicating how many constraints need to be satisfied in order to lead to an acceptable model. Typically, one wants
        to set this to 1 but for highly constrained problems one can relax this to find an approximation of a model satisfying all the
        constraints.
    max_bumps : int
        maximum number of bumps without improving the current best model that is allowed before resetting the training to the best model.
    max_time_after_bump : int
        maximum number of epochs after a single bump in which there is no improvement to the best model.
    max_constant_perf : int
        maximum number of epochs in which there is no improve in performance compared to the best and all constraints reached an acceptable
        satisfaction ratio. In other words, the maximum number of epochs in which the model does not improve and there is no bump. This acts
        in the same way as early-stopping would impact the training procedure.
    max_restarts : int
        maximum number of restarts that can be performed during training.
    save_location_model : string
        path to file for saving the model weights of the best performing model during training.
    save_file_model : string
        Name of file containing the weights of the trained model.
    lower_bound_output_variables : tf.Tensor
        Tensor of shape=(1, output_dimension) and dtype=tf.float32 denoting the lower bound used for the normalization of each output
        variable.
    upper_bound_output_variables : tf.Tensor
        Tensor of shape=(1, output_dimension) and dtype=tf.float32 denoting the upper bound used for the normalization of each output
        variable.
    lower_bound_input_variables : tf.Tensor
        Tensor of shape=(1, input_dimension) and dtype=tf.float32 denoting the lower bound used for the normalization of each input
        variable.
    upper_bound_input_variables : tf.Tensor
        Tensor of shape=(1, input_dimension) and dtype=tf.float32 denoting the upper bound used for the normalization of each input
        variable.

    Returns
    ----------
    model : tf.keras.Model
        best performing model obtained during training. This is not necessarily the last model obtained during training model.
    train_loss_results : list[tf.Tensor]
        List containing the value of the loss function of the training set for each epoch.
    train_metric_results : list[tf.Tensor]
        List containing the value of the metric of the training set for each epoch.
    train_satisfaction_results : list[tf.Tensor]
        List containing the satisfaction ratio of the training set for each epoch.
    val_loss_results : list[tf.Tensor]
        List containing the value of the loss function of the validation set for each epoch.
    val_metric_results : list[tf.Tensor]
        List containing the value of the metric of the validation set for each epoch.
    val_satisfaction_results : list[tf.Tensor]
        List containing the satisfaction ratio of the validation set for each epoch.
    timing : list[float]
        List containing the end time of each epoch.
    """
    ### check for existence of directory for saving model weights
    if not os.path.isdir(save_location_model):
        os.makedirs(save_location_model)
    ### initialize constant tensors for computing satisfaction ratio of constraints
    # tensors for train set
    ind_tensors_train = initialize_indicator_tensors(batch_size=train_batch_size,
                                                     output_dim=output_dim,
                                                     neur_act=neur_act,
                                                     list_bound_constraints=list_bound_constraints)

    if not train_number % train_batch_size == 0:  # for the final batch that is smaller than the others, the constant tensors containing
        # bounds and indicators need to be smaller
        new_train_batch_size = train_number % train_batch_size
        ind_tensors_train_last = initialize_indicator_tensors(batch_size=new_train_batch_size,
                                                              output_dim=output_dim,
                                                              neur_act=neur_act,
                                                              list_bound_constraints=list_bound_constraints)

    # tensors for validation set (only when the user gave a validation set)
    ind_tensors_val = initialize_indicator_tensors(batch_size=val_batch_size,
                                                   output_dim=output_dim,
                                                   neur_act=neur_act,
                                                   list_bound_constraints=list_bound_constraints)

    if not val_number % val_batch_size == 0:
        new_val_batch_size = val_number % val_batch_size
        ind_tensors_val_last = initialize_indicator_tensors(batch_size=new_val_batch_size,
                                                            output_dim=output_dim,
                                                            neur_act=neur_act,
                                                            list_bound_constraints=list_bound_constraints)

    lower_bound_output_variables = tf.reshape(tf.convert_to_tensor(lower_bound_output_variables), [1, output_dim])
    upper_bound_output_variables = tf.reshape(tf.convert_to_tensor(upper_bound_output_variables), [1, output_dim])
    lower_bound_input_variables = tf.reshape(tf.convert_to_tensor(lower_bound_input_variables), [1, 1])
    upper_bound_input_variables = tf.reshape(tf.convert_to_tensor(upper_bound_input_variables), [1, 1])
    undo_norm_direction = tf.math.subtract(upper_bound_output_variables, lower_bound_output_variables)

    ### define variables that are used throughout the training procedure
    # initialize lists for recording results
    train_loss_results = []
    train_metric_results = []
    train_satisfaction_results = []

    val_loss_results = []
    val_metric_results = []
    val_satisfaction_results = []

    # epoch numbers
    epoch = 1

    # define learning_rates
    learning_rate_loss = learning_rate
    list_lr_loss = [learning_rate_loss]

    # initialize metrics for recording values of loss function, additional metrics and satisfaction ratio for each batch in an epoch
    epoch_train_loss_average = tf.keras.metrics.Mean()
    epoch_train_metric_average = tf.keras.metrics.Mean()
    epoch_train_satisfaction_average = tf.keras.metrics.Mean()

    epoch_val_loss_average = tf.keras.metrics.Mean()
    epoch_val_metric_average = tf.keras.metrics.Mean()
    epoch_val_satisfaction_average = tf.keras.metrics.Mean()

    # initialize previous values of loss function, additional metrics and satisfaction ratio
    prev_train_sat_average = 10 ** 3
    prev_train_loss_average = 10 ** 3
    prev_train_met_average = 10 ** 3

    prev_val_sat_average = 10 ** 3
    prev_val_loss_average = 10 ** 3
    prev_val_met_average = 10 ** 3

    best_loss_average = 10 ** 3
    best_metric_average = 10 ** 3

    constant_loss = 1

    timing = []  # initialize list of times of the epochs of the training process

    timing.append(time.time())

    ### start of training procedure
    while epoch <= max_epochs:
        for x, y in train_dataset:
            current_batch_size = np.size(x, 0)  # determine the size of the batch
            if current_batch_size == train_batch_size:
                # optimize model
                gradient_loss, train_loss_value, train_metric_value, train_satisfaction_ratio \
                    = compute_fuzzy_grad(model=model,
                                         inputs=x,
                                         targets=y,
                                         batch_size=current_batch_size,
                                         output_dim=output_dim,
                                         neur_act=neur_act,
                                         bound_sm_eq=ind_tensors_train[0],
                                         bound_sm_st=ind_tensors_train[1],
                                         bound_la_eq=ind_tensors_train[2],
                                         bound_la_st=ind_tensors_train[3],
                                         ind_sm_eq=ind_tensors_train[4],
                                         ind_sm_st=ind_tensors_train[5],
                                         ind_la_eq=ind_tensors_train[6],
                                         ind_la_st=ind_tensors_train[7],
                                         signed_indicator_food_expense=ind_tensors_train[8],
                                         signed_indicator_income=ind_tensors_train[9],
                                         lower_bound_output_variables=lower_bound_output_variables,
                                         upper_bound_output_variables=upper_bound_output_variables,
                                         lower_bound_input_variables=lower_bound_input_variables,
                                         upper_bound_input_variables=upper_bound_input_variables,
                                         undo_norm_direction=undo_norm_direction)

                optimizer.apply_gradients(zip(gradient_loss, model.trainable_variables))
            else:  # use the constant tensors and indicator tensors with size corresponding to the last (and smaller) batch
                # optimize model
                gradient_loss, train_loss_value, train_metric_value, train_satisfaction_ratio \
                    = compute_fuzzy_grad(model=model,
                                         inputs=x,
                                         targets=y,
                                         batch_size=current_batch_size,
                                         output_dim=output_dim,
                                         neur_act=neur_act,
                                         bound_sm_eq=ind_tensors_train_last[0],
                                         bound_sm_st=ind_tensors_train_last[1],
                                         bound_la_eq=ind_tensors_train_last[2],
                                         bound_la_st=ind_tensors_train_last[3],
                                         ind_sm_eq=ind_tensors_train_last[4],
                                         ind_sm_st=ind_tensors_train_last[5],
                                         ind_la_eq=ind_tensors_train_last[6],
                                         ind_la_st=ind_tensors_train_last[7],
                                         signed_indicator_food_expense=ind_tensors_train_last[8],
                                         signed_indicator_income=ind_tensors_train_last[9],
                                         lower_bound_output_variables=lower_bound_output_variables,
                                         upper_bound_output_variables=upper_bound_output_variables,
                                         lower_bound_input_variables=lower_bound_input_variables,
                                         upper_bound_input_variables=upper_bound_input_variables,
                                         undo_norm_direction=undo_norm_direction)
                optimizer.apply_gradients(zip(gradient_loss, model.trainable_variables))

            # track progress from training batch
            epoch_train_loss_average.update_state(values=train_loss_value, sample_weight=current_batch_size/train_batch_size)
            epoch_train_metric_average.update_state(values=train_metric_value, sample_weight=current_batch_size/train_batch_size)
            epoch_train_satisfaction_average.update_state(values=train_satisfaction_ratio, sample_weight=current_batch_size/train_batch_size)

        # track progress from epoch
        train_loss_results.append(epoch_train_loss_average.result())
        train_metric_results.append(epoch_train_metric_average.result())
        train_satisfaction_results.append(epoch_train_satisfaction_average.result())

        # check performance on validation set
        for x, y in val_dataset:
            current_batch_size = np.size(x, 0)  # determine the size of the batch
            if current_batch_size == val_batch_size:
                val_loss_value, val_metric_value, val_sat_ratio = \
                    validation_check(model=model,
                                     inputs=x,
                                     targets=y,
                                     batch_size=current_batch_size,
                                     output_dim=output_dim,
                                     neur_act=neur_act,
                                     bound_sm_eq=ind_tensors_val[0],
                                     bound_sm_st=ind_tensors_val[1],
                                     bound_la_eq=ind_tensors_val[2],
                                     bound_la_st=ind_tensors_val[3],
                                     ind_sm_eq=ind_tensors_val[4],
                                     ind_sm_st=ind_tensors_val[5],
                                     ind_la_eq=ind_tensors_val[6],
                                     ind_la_st=ind_tensors_val[7],
                                     signed_indicator_food_expense=ind_tensors_val[8],
                                     signed_indicator_income=ind_tensors_val[9],
                                     lower_bound_output_variables=lower_bound_output_variables,
                                     upper_bound_output_variables=upper_bound_output_variables,
                                     lower_bound_input_variables=lower_bound_input_variables,
                                     upper_bound_input_variables=upper_bound_input_variables,
                                     undo_norm_direction=undo_norm_direction)

            else:
                val_loss_value, val_metric_value, val_sat_ratio = \
                    validation_check(model=model,
                                     inputs=x,
                                     targets=y,
                                     batch_size=current_batch_size,
                                     output_dim=output_dim,
                                     neur_act=neur_act,
                                     bound_sm_eq=ind_tensors_val_last[0],
                                     bound_sm_st=ind_tensors_val_last[1],
                                     bound_la_eq=ind_tensors_val_last[2],
                                     bound_la_st=ind_tensors_val_last[3],
                                     ind_sm_eq=ind_tensors_val_last[4],
                                     ind_sm_st=ind_tensors_val_last[5],
                                     ind_la_eq=ind_tensors_val_last[6],
                                     ind_la_st=ind_tensors_val_last[7],
                                     signed_indicator_food_expense=ind_tensors_val_last[8],
                                     signed_indicator_income=ind_tensors_val_last[9],
                                     lower_bound_output_variables=lower_bound_output_variables,
                                     upper_bound_output_variables=upper_bound_output_variables,
                                     lower_bound_input_variables=lower_bound_input_variables,
                                     upper_bound_input_variables=upper_bound_input_variables,
                                     undo_norm_direction=undo_norm_direction
                                     )

            # track progress from validation batch
            epoch_val_loss_average.update_state(values=val_loss_value, sample_weight=current_batch_size / val_batch_size)
            epoch_val_metric_average.update_state(values=val_metric_value, sample_weight=current_batch_size / val_batch_size)
            epoch_val_satisfaction_average.update_state(values=val_sat_ratio, sample_weight=current_batch_size / val_batch_size)

        # track progress from epoch
        val_loss_results.append(epoch_val_loss_average.result())
        val_metric_results.append(epoch_val_metric_average.result())
        val_satisfaction_results.append(epoch_val_satisfaction_average.result())

        print("Epoch {:03d}: Loss: {:.8f}, Mean absolute error: {:.8f}, Satisfaction rate val: {:.8%}, Satisfaction rate train: {:.8%}.".format(epoch,
                                                                                                                                                epoch_val_loss_average.result(),
                                                                                                                                                epoch_val_metric_average.result(),
                                                                                                                                                epoch_val_satisfaction_average.result(),
                                                                                                                                                epoch_train_satisfaction_average.result()))

        # save model if its performance is better
        if epoch_val_loss_average.result() < best_loss_average:
            best_loss_average = epoch_val_loss_average.result()
            best_metric_average = epoch_val_metric_average.result()
            print("Saving model.")
            model.save_weights(filepath=save_location_model + save_file_model)
            best_learning_rate = learning_rate_loss

        if epoch_val_loss_average.result() > prev_val_loss_average:
            constant_loss += 1
        else:
            constant_loss = 1

        if constant_loss % 15 == 0:
            learning_rate_loss = learning_rate_loss / 2
            print('New learning rate: ' + str(learning_rate_loss) + '. Reason: constant performance on validation set.')
            optimizer.lr.assign(learning_rate_loss)

        if constant_loss == max_constant_perf:
            print('Too long constant performance on validation set. Perform early stopping.')
            model.load_weights(filepath=save_location_model + save_file_model)
            break

        # update the previous best parameters or overwrite when learning rate is changed
        if epoch_train_satisfaction_average.result() < prev_train_sat_average:
            prev_train_sat_average = epoch_train_satisfaction_average.result()
        if epoch_train_loss_average.result() < prev_train_loss_average:
            prev_train_loss_average = epoch_train_loss_average.result()
        if epoch_train_metric_average.result() < prev_train_met_average:
            prev_train_met_average = epoch_train_metric_average.result()

        if epoch_val_satisfaction_average.result() < prev_val_sat_average:
            prev_val_sat_average = epoch_val_satisfaction_average.result()
        if epoch_val_loss_average.result() < prev_val_loss_average:
            prev_val_loss_average = epoch_val_loss_average.result()
        if epoch_val_metric_average.result() < prev_val_met_average:
            prev_val_met_average = epoch_val_metric_average.result()

        list_lr_loss.append(learning_rate_loss)

        # reset metric states
        epoch_train_loss_average.reset_states()
        epoch_train_metric_average.reset_states()
        epoch_train_satisfaction_average.reset_states()

        epoch_val_loss_average.reset_states()
        epoch_val_metric_average.reset_states()
        epoch_val_satisfaction_average.reset_states()

        epoch += 1

        timing.append(time.time())

    model.load_weights(filepath=save_location_model + save_file_model)

    print('----------------TRAINING-IS-DONE----------------------')
    print('Total number of epochs is ' + str(epoch) + '.')
    print('Best validation loss is ' + str(best_loss_average.numpy()) + '.')
    print('Best metric loss is ' + str(best_metric_average.numpy()) + '.')

    return model, train_loss_results, train_metric_results, train_satisfaction_results, val_loss_results, val_metric_results, val_satisfaction_results, epoch, list_lr_loss, timing


def test_model(model, input_dim, output_dim, neur_act, list_bound_constraints, test_number, test_dataset, test_batch_size,
               acceptable_sat_ratio, lower_bound_output_variables, upper_bound_output_variables, lower_bound_input_variables,
               upper_bound_input_variables):
    """Test the given model on a test set for loss value, metric value and satisfaction ratio.

    Parameters
    ----------
    model : tf.keras.Model
        model to be evaluated on test set.
    input_dim : int
        integer denoting the input dimension of the model.
    output_dim : int
        integer denoting the output dimension of the model.
    neur_act : list[int]
        list of integers denoting the number of neurons and activation functions in the layers of the network.
    list_bound_constraints : list[list[tf.Tensor]]
        list denoting the bound constraints.
    test_number : int
        number of examples in the test set.
    test_dataset : tf.data.Dataset
        test set
    test_batch_size : int
        batch-size used to do the testing of the model
    acceptable_sat_ratio : float
        the acceptable satisfaction ratio that was used during training.
    lower_bound_output_variables : tf.Tensor
        Tensor of shape=(1, output_dimension) and dtype=tf.float32 denoting the lower bound used for the normalization of each output
        variable.
    upper_bound_output_variables : tf.Tensor
        Tensor of shape=(1, output_dimension) and dtype=tf.float32 denoting the upper bound used for the normalization of each output
        variable.
    lower_bound_input_variables : tf.Tensor
        Tensor of shape=(1, input_dimension) and dtype=tf.float32 denoting the lower bound used for the normalization of each input
        variable.
    upper_bound_input_variables : tf.Tensor
        Tensor of shape=(1, input_dimension) and dtype=tf.float32 denoting the upper bound used for the normalization of each input
        variable.

    Returns
    ----------
    loss_result : float
        float denoting the loss value averaged over the whole test set
    metric_result : float
        float denoting the value of the metric averaged over the whole test set
    sat_ratio_result : float
        float in [0,1] denoting the satisfaction ratio (interpreted as a percentage) over the whole test set
    relative_sat_ratio_result : float
        float denoting the relative difference with respect to the acceptable satisfaction ratio used during training.

    """
    ### initialize constant tensors for computing satisfaction ratio of constraints
    # tensors for train set
    ind_tensors_test = initialize_indicator_tensors(batch_size=test_batch_size,
                                                    output_dim=output_dim,
                                                    neur_act=neur_act,
                                                    list_bound_constraints=list_bound_constraints)

    if not test_number % test_batch_size == 0:  # for the final batch that is smaller than the others, the constant tensors containing
        # bounds and indicators need to be smaller
        new_test_batch_size = test_number % test_batch_size
        ind_tensors_test_last = initialize_indicator_tensors(batch_size=new_test_batch_size,
                                                             output_dim=output_dim,
                                                             neur_act=neur_act,
                                                             list_bound_constraints=list_bound_constraints)

    lower_bound_output_variables = tf.reshape(tf.convert_to_tensor(lower_bound_output_variables), [1, output_dim])
    upper_bound_output_variables = tf.reshape(tf.convert_to_tensor(upper_bound_output_variables), [1, output_dim])
    lower_bound_input_variables = tf.reshape(tf.convert_to_tensor(lower_bound_input_variables), [1, 1])
    upper_bound_input_variables = tf.reshape(tf.convert_to_tensor(upper_bound_input_variables), [1, 1])
    undo_norm_direction = tf.math.subtract(upper_bound_output_variables, lower_bound_output_variables)

    # initialize metrics for recording values of loss function, additional metrics and satisfaction ratio for each batch in the test set
    test_loss_average = tf.keras.metrics.Mean()
    test_metric_average = tf.keras.metrics.Mean()
    test_satisfaction_average = tf.keras.metrics.Mean()

    # check performance on test set
    for x, y in test_dataset:
        current_batch_size = np.size(x, 0)  # determine the size of the batch
        if current_batch_size == test_batch_size:
            test_loss_value, test_metric_value, test_sat_ratio = \
                validation_check(model=model,
                                 inputs=x,
                                 targets=y,
                                 batch_size=current_batch_size,
                                 output_dim=output_dim,
                                 neur_act=neur_act,
                                 bound_sm_eq=ind_tensors_test[0],
                                 bound_sm_st=ind_tensors_test[1],
                                 bound_la_eq=ind_tensors_test[2],
                                 bound_la_st=ind_tensors_test[3],
                                 ind_sm_eq=ind_tensors_test[4],
                                 ind_sm_st=ind_tensors_test[5],
                                 ind_la_eq=ind_tensors_test[6],
                                 ind_la_st=ind_tensors_test[7],
                                 signed_indicator_food_expense=ind_tensors_test[8],
                                 signed_indicator_income=ind_tensors_test[9],
                                 lower_bound_output_variables=lower_bound_output_variables,
                                 upper_bound_output_variables=upper_bound_output_variables,
                                 lower_bound_input_variables=lower_bound_input_variables,
                                 upper_bound_input_variables=upper_bound_input_variables,
                                 undo_norm_direction=undo_norm_direction)

        else:
            test_loss_value, test_metric_value, test_sat_ratio = \
                validation_check(model=model,
                                 inputs=x,
                                 targets=y,
                                 batch_size=current_batch_size,
                                 output_dim=output_dim,
                                 neur_act=neur_act,
                                 bound_sm_eq=ind_tensors_test_last[0],
                                 bound_sm_st=ind_tensors_test_last[1],
                                 bound_la_eq=ind_tensors_test_last[2],
                                 bound_la_st=ind_tensors_test_last[3],
                                 ind_sm_eq=ind_tensors_test_last[4],
                                 ind_sm_st=ind_tensors_test_last[5],
                                 ind_la_eq=ind_tensors_test_last[6],
                                 ind_la_st=ind_tensors_test_last[7],
                                 signed_indicator_food_expense=ind_tensors_test_last[8],
                                 signed_indicator_income=ind_tensors_test_last[9],
                                 lower_bound_output_variables=lower_bound_output_variables,
                                 upper_bound_output_variables=upper_bound_output_variables,
                                 lower_bound_input_variables=lower_bound_input_variables,
                                 upper_bound_input_variables=upper_bound_input_variables,
                                 undo_norm_direction=undo_norm_direction)

        # track progress from test batch
        test_loss_average.update_state(values=test_loss_value, sample_weight=current_batch_size/test_batch_size)
        test_metric_average.update_state(values=test_metric_value, sample_weight=current_batch_size/test_batch_size)
        test_satisfaction_average.update_state(values=test_sat_ratio, sample_weight=current_batch_size/test_batch_size)

    loss_result = test_loss_average.result().numpy()
    metric_result = test_metric_average.result().numpy()
    sat_ratio_result = test_satisfaction_average.result().numpy()

    relative_sat_ratio_result = (sat_ratio_result-acceptable_sat_ratio)/acceptable_sat_ratio

    return loss_result, metric_result, sat_ratio_result, relative_sat_ratio_result

