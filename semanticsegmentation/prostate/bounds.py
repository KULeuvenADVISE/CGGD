#!/usr/bin/env python3.7

# from itertools import repeat
from typing import Any, Callable, List, Tuple

import torch
from torch import Tensor

from utils import eq, sset
from utils import BoxCoords, binary2boxcoords, boxcoords2masks_bounds


class BoxPriorBounds():
    def __init__(self, **kwargs):
        self.d: int = kwargs['d']
        self.idc: List[int] = kwargs["idc"]

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, box_targets: Tensor) -> List[Tuple[Tensor, Tensor]]:
        K: int
        W: int
        H: int
        K, W, H = box_targets.shape
        assert sset(box_targets, [0, 1])

        # Because computing the boxes on the background class, then discarding it, would destroy the memory
        boxes_per_class: List[List[BoxCoords]]
        boxes_per_class = [binary2boxcoords(box_targets[k]) if k in self.idc else [] for k in range(K)]

        res: List[Tuple[Tensor, Tensor]]
        res = [boxcoords2masks_bounds(boxes, (W, H), self.d) for boxes in boxes_per_class]

        # Some images won't have any bounding box, so their length can be > 0
        # But make sure that the background classes do not have any result
        assert all(res[k][0].shape[0] == 0 if (k not in self.idc) else True for k in range(K))

        return res


class BoxBounds():
    def __init__(self, **kwargs):
        self.margins: Tensor = torch.Tensor(kwargs['margins'])
        assert len(self.margins) == 2
        assert self.margins[0] <= self.margins[1]

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        c = len(weak_target)
        box_sizes: Tensor = torch.einsum("cwh->c", weak_target)[..., None].type(torch.float32)

        bounds: Tensor = box_sizes * self.margins

        res = bounds[:, None, :]
        assert res.shape == (c, 1, 2)
        assert (res[..., 0] <= res[..., 1]).all()

        # exact_sizes: Tensor = torch.einsum("cwh->c", target).type(torch.float32)
        # assert (res[3, 0, 0] <= exact_sizes[3]).all(), (res[:, 0, 0], exact_sizes, box_sizes[..., 0])
        # assert (res[3, 0, 1] >= exact_sizes[3]).all(), (res[:, 0, 1], exact_sizes, box_sizes[..., 0])

        return res