#!/usr/env/bin python3.6

import pdb
from operator import add
from functools import reduce
from typing import List, Tuple

import torch
import numpy as np
from torch import Tensor, einsum

from utils import simplex, sset, probs2one_hot, one_hot, map_


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = target[:, self.idc, ...].type(torch.float32)

        loss = - einsum("bcwh,bcwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class NaivePenalty():
    """
    Implementation in the same fashion as the log-barrier
    """
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        self.C = len(self.idc)
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        def penalty(z: Tensor) -> Tensor:
            assert z.shape == ()

            return torch.max(torch.zeros_like(z), z)**2

        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        assert probs.shape == target.shape

        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        _, _, k, two = bounds.shape  # scalar or vector
        assert two == 2
        # assert k == 1  # Keep it simple for now
        value: Tensor = self.__fn__(probs[:, self.idc, ...])
        lower_b = bounds[:, self.idc, :, 0]
        upper_b = bounds[:, self.idc, :, 1]

        assert value.shape == (b, self.C, k), value.shape
        assert lower_b.shape == upper_b.shape == (b, self.C, k), lower_b.shape

        upper_z: Tensor = (value - upper_b).type(torch.float32).flatten()
        lower_z: Tensor = (lower_b - value).type(torch.float32).flatten()

        upper_penalty: Tensor = reduce(add, (penalty(e) for e in upper_z))
        lower_penalty: Tensor = reduce(add, (penalty(e) for e in lower_z))

        res: Tensor = upper_penalty + lower_penalty

        loss: Tensor = res.sum() / (w * h)
        assert loss.requires_grad == probs.requires_grad  # Handle the case for validation

        return loss


class LogBarrierLoss():
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        self.C = len(self.idc)
        self.t: float = kwargs["t"]
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        def log_barrier(z: Tensor) -> Tensor:
            assert z.shape == ()

            if z <= - 1 / self.t**2:
                return - torch.log(-z) / self.t
            else:
                return self.t * z + -np.log(1 / (self.t**2)) / self.t + 1 / self.t

        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        assert probs.shape == target.shape

        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        _, _, k, two = bounds.shape  # scalar or vector
        assert two == 2
        # assert k == 1  # Keep it simple for now
        value: Tensor = self.__fn__(probs[:, self.idc, ...])
        lower_b = bounds[:, self.idc, :, 0]
        upper_b = bounds[:, self.idc, :, 1]

        assert value.shape == (b, self.C, k), value.shape
        assert lower_b.shape == upper_b.shape == (b, self.C, k), lower_b.shape

        upper_z: Tensor = (value - upper_b).type(torch.float32).flatten()
        lower_z: Tensor = (lower_b - value).type(torch.float32).flatten()

        upper_barrier: Tensor = reduce(add, (log_barrier(e) for e in upper_z))
        lower_barrier: Tensor = reduce(add, (log_barrier(e) for e in lower_z))

        res: Tensor = upper_barrier + lower_barrier

        loss: Tensor = res.sum() / (w * h)
        assert loss.requires_grad == probs.requires_grad  # Handle the case for validation

        return loss


class BoxPrior():
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]

        self.t: float = kwargs["t"]

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def barrier(self, z: Tensor) -> Tensor:
        assert z.shape == ()

        if z <= - 1 / self.t**2:
            return - torch.log(-z) / self.t
        else:
            return self.t * z + -np.log(1 / (self.t**2)) / self.t + 1 / self.t

    def __call__(self, probs: Tensor, _: Tensor, __: Tensor,
                 box_prior: List[List[Tuple[Tensor, Tensor]]]) -> Tensor:
        assert simplex(probs)

        B: int = probs.shape[0]
        assert len(box_prior) == B

        sublosses = []
        for b in range(B):
            for k in self.idc:
                masks, bounds = box_prior[b][k]

                sizes: Tensor = einsum('wh,nwh->n', probs[b, k], masks)

                assert sizes.shape == bounds.shape == (masks.shape[0],), (sizes.shape, bounds.shape, masks.shape)
                shifted: Tensor = bounds - sizes

                init = torch.zeros((), dtype=torch.float32, requires_grad=probs.requires_grad, device=probs.device)
                sublosses.append(reduce(add, (self.barrier(v) for v in shifted), init))

        loss: Tensor = reduce(add, sublosses)

        assert loss.dtype == torch.float32
        assert loss.shape == (), loss.shape

        return loss


class BoxPriorCGGD():
    def __init__(self, device, **kwargs):
        self.idc: List[int] = kwargs["idc"]

        self.t: float = kwargs["t"]
        self.device = device

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, _: Tensor, __: Tensor,
                 box_prior: List[List[Tuple[Tensor, Tensor]]]) -> Tuple[List, List]:
        assert simplex(probs)

        B: int = probs.shape[0]
        assert len(box_prior) == B

        sublosses = []
        directions = []
        for b in range(B):
            for k in self.idc:
                masks, bounds = box_prior[b][k]

                sizes: Tensor = einsum('wh,nwh->n', probs[b, k], masks)

                assert sizes.shape == bounds.shape == (masks.shape[0],), (sizes.shape, bounds.shape, masks.shape)
                shifted: Tensor = torch.less_equal(sizes, bounds)

                direction: Tensor = torch.where(shifted, torch.tensor(-1, dtype=torch.float32, device=self.device), torch.tensor(0, dtype=torch.float32, device=self.device))

                sublosses.append(shifted)
                directions.append(direction)

        return sublosses, directions

