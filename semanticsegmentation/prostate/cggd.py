"""

:author: Quinten Van Baelen (2023)
:license: Apache License, Version 2.0, see LICENSE for details.

"""

import os
import torch
import numpy as np


def check_lower_bound(predictions, lower_bound, device):
    """Check which predictions satisfy the lower bound constraints and which do not.

    Parameters
    ----------
    predictions : torch.Tensor
        A torch.Tensor of shape=(batch_size, output_dim) and dType=torch.float32 containing the predictions of the network for the current batch.
    lower_bound : torch.Tensor
        A torch.Tensor of shape=(batch_size, output_dim) and dType=torch.float32 containing the lower bounds of the output variables of the network.
        If an output variable does not have a lower bound, then this value is arbitrary.
    device : torch.device
        A torch.device denoting which GPU should be used or the CPU.

    Returns
    ----------
    satisfied_lower_bound : torch.Tensor
        A torch.Tensor of shape=(batch_size, output_dim) and dType=torch.bool indicating which predictions satisfy the lower bound constraint when it
        has one.
    unsatisfied_lower_bound : torch.Tensor
        A torch.Tensor of shape=(batch_size, output_dim) and dType=torch.bool indicating which predictions do not satisfy the lower bound constraint
        when it has one. This should be used to remove the direction of the constraints for the examples that satisfy the constraints.
    number_satisfied_lower_bound : torch.Tensor
        A torch.Tensor of shape=(,) and dType=torch.float32 denoting the number of examples that satisfy a lower bound constraint. This is counted
        with multiplicity, that is, if there are two lower bound constraints on the network and the predictions of the network satisfies both then
        this counts as 2 in the sum. In case only a single lower bound constraint would be satisfied, then this counts as 1 in the sum. The sum is
        taken over all examples in the batch.
    number_lower_bound : torch.Tensor
        A torch.Tensor of shape=(,) and dType=torch.float32 denoting the number of lower bound constraints for the batch. This is counted with
        multiplicity, that is, if the network has a lower bound constraint on two output variables, then this number is equal to 2*batch_size.
    """
    satisfied_lower_bound = torch.less_equal(lower_bound, predictions).to(device)
    unsatisfied_lower_bound = torch.logical_not(satisfied_lower_bound)

    number_satisfied_lower_bound = torch.sum(input=torch.where(satisfied_lower_bound,
                                                               torch.tensor(1, dtype=torch.float32, device=device),
                                                               torch.tensor(0, dtype=torch.float32, device=device))).to(device)

    number_lower_bound = torch.tensor(torch.numel(predictions), dtype=torch.float32, device=device)

    return satisfied_lower_bound, unsatisfied_lower_bound, number_satisfied_lower_bound, number_lower_bound


def check_upper_bound(predictions, upper_bound, device):
    """Check which predictions satisfy the upper bound constraints and which do not.

    Parameters
    ----------
    predictions : torch.Tensor
        A torch.Tensor of shape=(batch_size, output_dim) and dType=torch.float32 containing the predictions of the network for the current batch.
    upper_bound : torch.Tensor
        A torch.Tensor of shape=(batch_size, output_dim) and dType=torch.float32 containing the upper bounds of the output variables of the network.
        If an output variable does not have a upper bound, then this value is arbitrary.
    device : torch.device
        A torch.device denoting which GPU should be used or the CPU.

    Returns
    ----------
    satisfied_upper_bound : torch.Tensor
        A torch.Tensor of shape=(batch_size, output_dim) and dType=torch.bool indicating which predictions satisfy the upper bound constraint when it
        has one.
    unsatisfied_upper_bound : torch.Tensor
        A torch.Tensor of shape=(batch_size, output_dim) and dType=torch.bool indicating which predictions do not satisfy the upper bound constraint
        when it has one. This should be used to remove the direction of the constraints for the examples that satisfy the constraints.
    number_satisfied_upper_bound : torch.Tensor
        A torch.Tensor of shape=(,) and dType=torch.float32 denoting the number of examples that satisfy a upper bound constraint. This is counted
        with multiplicity, that is, if there are two upper bound constraints on the network and the predictions of the network satisfies both then
        this counts as 2 in the sum. In case only a single upper bound constraint would be satisfied, then this counts as 1 in the sum. The sum is
        taken over all examples in the batch.
    number_upper_bound : torch.Tensor
        A torch.Tensor of shape=(,) and dType=torch.float32 denoting the number of upper bound constraints for the batch. This is counted with
        multiplicity, that is, if the network has a upper bound constraint on two output variables, then this number is equal to 2*batch_size.
    """
    satisfied_upper_bound = torch.greater_equal(upper_bound, predictions).to(device)
    unsatisfied_upper_bound = torch.logical_not(satisfied_upper_bound)

    number_satisfied_upper_bound = torch.sum(input=torch.where(satisfied_upper_bound,
                                                               torch.tensor(1, dtype=torch.float32, device=device),
                                                               torch.tensor(0, dtype=torch.float32, device=device))).to(device)

    number_upper_bound = torch.tensor(torch.numel(predictions), dtype=torch.float32, device=device)

    return satisfied_upper_bound, unsatisfied_upper_bound, number_satisfied_upper_bound, number_upper_bound


def compute_direction_lower_bound(unsatisfied_lower_bound, device):
    """Compute the direction for the lower bounds when they are not satisfied.

    Parameters
    ----------
    unsatisfied_lower_bound : torch.Tensor
        A torch.Tensor of shape=(batch_size, output_dim) and dType=torch.bool denoting if the lower bound is not satisfied (True) or satisfied (False)
        for each output variable. If the output variable does not have a lower bound, then the corresponding value is False.
    device : torch.device
        A torch.device denoting which GPU should be used or the CPU.

    Returns
    ----------
    direction_lower_bound_unsatisfied : torch.Tensor
        A torch.Tensor of shape=(batch_size, output_dim) and dType=torch.float32 denoting the direction of the constraints for the output variables
        that do not satisfy the corresponding lower bound.
    """
    direction_lower_bound_unsatisfied = torch.where(unsatisfied_lower_bound,
                                                    torch.tensor(-1, dtype=torch.float32, device=device),
                                                    torch.tensor(0, dtype=torch.float32, device=device)).to(device)
    return direction_lower_bound_unsatisfied


def compute_direction_upper_bound(unsatisfied_upper_bound, device):
    """Compute the direction for the upper bounds when they are not satisfied.

    Parameters
    ----------
    unsatisfied_upper_bound : torch.Tensor
        A torch.Tensor of shape=(batch_size, output_dim) and dType=torch.bool denoting if the upper bound is not satisfied (True) or satisfied (False)
        for each output variable. If the output variable does not have a lower bound, then the corresponding value is False.
    device : torch.device
        A torch.device denoting which GPU should be used or the CPU.

    Returns
    ----------
    direction_upper_bound_unsatisfied : torch.Tensor
        A torch.Tensor of shape=(batch_size, output_dim) and dType=torch.float32 denoting the direction of the constraints for the output variables
        that do not satisfy the corresponding upper bound.
    """
    direction_upper_bound_unsatisfied = torch.where(unsatisfied_upper_bound,
                                                    torch.tensor(1, dtype=torch.float32, device=device),
                                                    torch.tensor(0, dtype=torch.float32, device=device)).to(device)
    return direction_upper_bound_unsatisfied

