"""

This script is largely based on the script main.py in https://github.com/LIVIAETS/extended_logbarrier.

The modifications are as follows:
    - 6 new arguments in the parser (see lines 998-1004)
    - the function do_epoch_cggd_box_prior_box_size_residual_unet
    - the function run is adjusted to support selecting the function do_epoch_cggd_box_prior_box_size_residual_unet. This function trains the network using CGGD with the box prior
    and the box size constraint.

"""


#!/usr/bin/env python3.7

import argparse
import warnings
import itertools
from pathlib import Path
from functools import reduce
from operator import add, itemgetter
from shutil import copytree, rmtree
from typing import Any, Callable, Dict, List, Tuple, Optional, Union, cast

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import Tensor, einsum
from torch.utils.data import DataLoader

from dataloader import get_loaders
from utils import map_
from utils import depth
from utils import inter_sum, union_sum
from utils import probs2one_hot, probs2class
from utils import dice_coef, save_images, tqdm_, hausdorff, iIoU, dice_batch
from utils import simplex

import cggd


def setup(args, n_class: int) -> Tuple[Any, Any, Any, List[List[Callable]], List[List[float]], Callable]:
    print("\n>>> Setting up")
    #cpu: bool = args.cpu or not torch.cuda.is_available()
    cpu: bool = not torch.cuda.is_available()
    #print(cpu)
    device = torch.device("cpu") if cpu else torch.device("cuda:" + args.gpu_number)
    #device = torch.device("cpu") if cpu else torch.device("cuda:3")
    #print(device)
    torch.manual_seed(args.network_seed)

    if args.weights:
        if cpu:
            net = torch.load(args.weights, map_location='cpu')
        else:
            net = torch.load(args.weights)
        print(f">> Restored weights from {args.weights} successfully.")
    else:
        net_class = getattr(__import__('networks'), args.network)
        net = net_class(args.modalities, n_class).to(device)
        net.init_weights()
    net.to(device)

    optimizer: Any  # disable an error for the optmizer (ADAM and SGD not same type)
    if args.use_sgd:
        optimizer = torch.optim.SGD(net.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=args.l_rate, betas=(0.9, 0.99), amsgrad=False)

    # print(args.losses)
    list_losses = eval(args.losses)
    if depth(list_losses) == 1:  # For compatibility reasons, avoid changing all the previous configuration files
        list_losses = [list_losses]

    nd: str = "whd" if args.three_d else "wh"

    loss_fns: List[List[Callable]] = []
    for i, losses in enumerate(list_losses):
        print(f">> {i}th list of losses: {losses}")
        tmp: List[Callable] = []
        for loss_name, loss_params, _, _, fn, _ in losses:
            loss_class = getattr(__import__('losses'), loss_name)
            tmp.append(loss_class(**loss_params, fn=fn, nd=nd))
        loss_fns.append(tmp)

    loss_weights: List[List[float]] = [map_(itemgetter(5), losses) for losses in list_losses]

    scheduler = getattr(__import__('scheduler'), args.scheduler)(**eval(args.scheduler_params))

    return net, optimizer, device, loss_fns, loss_weights, scheduler


def do_epoch(mode: str, net: Any, device: Any, loaders: List[DataLoader], epc: int,
             list_loss_fns: List[List[Callable]], list_loss_weights: List[List[float]], K: int,
             savedir: str = "", optimizer: Any = None,
             metric_axis: List[int] = [1], compute_hausdorff: bool = False, compute_miou: bool = False,
             compute_3d_dice: bool = False,
             temperature: float = 1) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Tensor, Tensor]:
    assert mode in ["train", "val"]

    if mode == "train":
        net.train()
        desc = f">> Training   ({epc})"
    elif mode == "val":
        net.eval()
        desc = f">> Validation ({epc})"

    total_iteration: int = sum(len(loader) for loader in loaders)  # U
    total_images: int = sum(len(loader.dataset) for loader in loaders)  # D
    n_loss: int = max(map(len, list_loss_fns))

    all_dices: Tensor = torch.zeros((total_images, K), dtype=torch.float32, device=device)
    loss_log: Tensor = torch.zeros((total_iteration, n_loss), dtype=torch.float32, device=device)

    iiou_log: Optional[Tensor]
    intersections: Optional[Tensor]
    unions: Optional[Tensor]
    if compute_miou:
        iiou_log = torch.zeros((total_images, K), dtype=torch.float32, device=device)
        intersections = torch.zeros((total_images, K), dtype=torch.float32, device=device)
        unions = torch.zeros((total_images, K), dtype=torch.float32, device=device)
    else:
        iiou_log = None
        intersections = None
        unions = None

    number_con: Tensor = torch.zeros((1, 3), dtype=torch.float32, device=device)
    number_sat_con: Tensor = torch.zeros((1, 3), dtype=torch.float32, device=device)

    three_d_dices: Optional[Tensor]
    if compute_3d_dice:
        three_d_dices = torch.zeros((total_iteration, K), dtype=torch.float32, device=device)
    else:
        three_d_dices = None

    hausdorff_log: Optional[Tensor]
    if compute_hausdorff:
        hausdorff_log = torch.zeros((total_images, K), dtype=torch.float32, device=device)
    else:
        hausdorff_log = None

    few_axis: bool = len(metric_axis) <= 3

    done_img: int = 0
    done_batch: int = 0
    tq_iter = tqdm_(total=total_iteration, desc=desc)
    for i, (loader, loss_fns, loss_weights) in enumerate(zip(loaders, list_loss_fns, list_loss_weights)):
        for data in loader:
            image: Tensor = data["images"].to(device)
            target: Tensor = data["gt"].to(device)
            spacings: Tensor = data["spacings"]  # Keep that one on CPU
            assert not target.requires_grad
            labels: List[Tensor] = [e.to(device) for e in data["labels"]]
            bounds: List[Tensor] = [e.to(device) for e in data["bounds"]]
            box_priors: List[List[Tuple[Tensor, Tensor]]]  # one more level for the batch
            box_priors = [[(m.to(device), b.to(device)) for (m, b) in B] for B in data["box_priors"]]
            assert len(labels) == len(bounds)

            B, C, *_ = image.shape

            samplings: List[List[Tuple[slice]]] = data["samplings"]
            assert len(samplings) == B
            assert len(samplings[0][0]) == len(image[0, 0].shape), (samplings[0][0], image[0, 0].shape)

            probs_receptacle: Tensor = - torch.ones_like(target, dtype=torch.float32)  # -1 for unfilled
            mask_receptacle: Tensor = - torch.ones_like(target, dtype=torch.int32)  # -1 for unfilled

            # Use the sampling coordinates of the first batch item
            assert not (len(samplings[0]) > 1 and B > 1), samplings  # No subsampling if batch size > 1
            loss_sub_log: Tensor = torch.zeros((len(samplings[0]), len(loss_fns)), dtype=torch.float32, device=device)
            for k, sampling in enumerate(samplings[0]):
                img_sampling = [slice(0, B), slice(0, C)] + list(sampling)
                label_sampling = [slice(0, B), slice(0, K)] + list(sampling)
                assert len(img_sampling) == len(image.shape), (img_sampling, image.shape)
                sub_img = image[img_sampling]

                # Reset gradients
                if optimizer:
                    optimizer.zero_grad()

                # Forward
                pred_logits: Tensor = net(sub_img)
                pred_probs: Tensor = F.softmax(temperature * pred_logits, dim=1)
                predicted_mask: Tensor = probs2one_hot(pred_probs.detach())  # Used only for dice computation
                assert not predicted_mask.requires_grad

                probs_receptacle[label_sampling] = pred_probs[...]
                mask_receptacle[label_sampling] = predicted_mask[...]

                assert len(bounds) == len(loss_fns) == len(loss_weights) == len(labels)
                ziped = zip(loss_fns, labels, loss_weights, bounds)
                losses = [w * loss_fn(pred_probs, label[label_sampling], bound, box_priors) for loss_fn, label, w, bound in ziped]
                loss = reduce(add, losses)
                assert loss.shape == (), loss.shape

                with torch.no_grad():
                    ziped = zip(loss_fns, labels, loss_weights, bounds)

                    # it is assumed that first the box prior constraint is used.
                    current_loss_fn, current_label, _, current_bound = next(itertools.islice(ziped, len(loss_fns) - 2, None))
                    #value1: Tensor = current_loss_fn.__fn__(pred_probs[:, current_loss_fn.idc, ...])

                    #number_bounds1 = torch.Tensor(0, dtype=torch.float32, device=device)
                    number_bounds1 = torch.zeros((), dtype=torch.float32, device=device)
                    #number_satisfied_bounds1 = torch.Tensor(0, dtype=torch.float32, device=device)
                    number_satisfied_bounds1 = torch.zeros((), dtype=torch.float32, device=device)

                    current_B = pred_probs.shape[0]
                    for b in range(current_B):
                        for l in current_loss_fn.idc:
                            current_masks, current_bounds = box_priors[b][l]

                            current_sizes: Tensor = einsum('wh,nwh->n', pred_probs[b, l], current_masks)

                            assert current_sizes.shape == current_bounds.shape == (current_masks.shape[0],), (current_sizes.shape, current_bounds.shape, current_masks.shape)

                            init_con = torch.zeros((), dtype=torch.float32, requires_grad=False, device=device)
                            number_bounds1 = torch.add(number_bounds1, reduce(add, (torch.numel(current_bounds), init_con)))
                            satisfied_bounds = torch.where(torch.greater_equal(current_sizes, current_bounds), torch.tensor(1, dtype=torch.float32, device=device), torch.tensor(0, dtype=torch.float32, device=device)).to(device)
                            number_satisfied_bounds1 = torch.add(number_satisfied_bounds1, reduce(add, (torch.sum(satisfied_bounds), init_con)))

                    # it is assumed that the box bounds constraint is used second.
                    ziped = zip(loss_fns, labels, loss_weights, bounds)
                    current_loss_fn, current_label, _, current_bounds = next(itertools.islice(ziped, len(loss_fns) - 1, None))
                    value2: Tensor = current_loss_fn.__fn__(pred_probs[:, current_loss_fn.idc, ...])

                    #number_bounds2 = torch.Tensor(0, dtype=torch.float32, device=device)
                    #number_satisfied_bounds2 = torch.Tensor(0, dtype=torch.float32, device=device)

                    current_lower_b = current_bounds[:, current_loss_fn.idc, :, 0]
                    current_upper_b = current_bounds[:, current_loss_fn.idc, :, 1]

                    #number_bounds_lower2 = torch.Tensor(0, dtype=torch.float32, device=value.device)
                    number_bounds_lower2 = torch.zeros((), dtype=torch.float32, device=device)
                    #number_bounds_upper2 = torch.Tensor(0, dtype=torch.float32, device=value.device)
                    number_bounds_upper2 = torch.zeros((), dtype=torch.float32, device=device)
                    #number_satisfied_bounds_lower2 = torch.Tensor(0, dtype=torch.float32, device=value.device)
                    number_satisfied_bounds_lower2 = torch.zeros((), dtype=torch.float32, device=device)
                    #number_satisfied_bounds_upper2 = torch.Tensor(0, dtype=torch.float32, device=value.device)
                    number_satisfied_bounds_upper2 = torch.zeros((), dtype=torch.float32, device=device)

                    number_bounds_upper2 = torch.add(number_bounds_upper2, torch.numel(current_upper_b))
                    number_satisfied_bounds_upper2 = torch.add(number_satisfied_bounds_upper2, torch.sum(torch.where(torch.greater_equal(current_upper_b, value2), torch.tensor(1, dtype=torch.float32, device=device), torch.tensor(0, dtype=torch.float32, device=device))).to(device))
                    number_bounds_lower2 = torch.add(number_bounds_lower2, torch.numel(current_lower_b))
                    number_satisfied_bounds_lower2 = torch.add(number_satisfied_bounds_lower2, torch.sum(torch.where(torch.greater_equal(value2, current_lower_b), torch.tensor(1, dtype=torch.float32, device=device), torch.tensor(0, dtype=torch.float32, device=device))).to(device))

                # Backward
                if optimizer:
                    loss.backward()
                    optimizer.step()

                # Compute and log metrics
                for j in range(len(loss_fns)):
                    loss_sub_log[k, j] = losses[j].detach()

                number_con = torch.add(number_con, torch.reshape(torch.tensor([number_bounds1, number_bounds_lower2, number_bounds_upper2], dtype=torch.float32, device=number_con.device), [1, 3]))
                number_sat_con = torch.add(number_sat_con, torch.reshape(torch.tensor([number_satisfied_bounds1, number_satisfied_bounds_lower2, number_satisfied_bounds_upper2], dtype=torch.float32, device=number_sat_con.device), [1, 3]))

            reduced_loss_sublog: Tensor = loss_sub_log.sum(dim=0)
            assert reduced_loss_sublog.shape == (len(loss_fns),), (reduced_loss_sublog.shape, len(loss_fns))
            loss_log[done_batch, ...] = reduced_loss_sublog[...]
            del loss_sub_log

            sm_slice = slice(done_img, done_img + B)  # Values only for current batch

            dices: Tensor = dice_coef(mask_receptacle, target)
            assert dices.shape == (B, K), (dices.shape, B, K)
            all_dices[sm_slice, ...] = dices

            if compute_3d_dice:
                three_d_DSC: Tensor = dice_batch(mask_receptacle, target)
                assert three_d_DSC.shape == (K,)

                three_d_dices[done_batch] = three_d_DSC  # type: ignore

            if compute_hausdorff:
                hausdorff_res: Tensor
                try:
                    hausdorff_res = hausdorff(mask_receptacle, target, spacings)
                except RuntimeError:
                    hausdorff_res = torch.zeros((B, K), device=device)
                assert hausdorff_res.shape == (B, K)
                hausdorff_log[sm_slice] = hausdorff_res  # type: ignore
            if compute_miou:
                IoUs: Tensor = iIoU(mask_receptacle, target)
                assert IoUs.shape == (B, K), IoUs.shape
                iiou_log[sm_slice] = IoUs  # type: ignore
                intersections[sm_slice] = inter_sum(mask_receptacle, target)  # type: ignore
                unions[sm_slice] = union_sum(mask_receptacle, target)  # type: ignore

            # if False and target[0, 1].sum() > 0:  # Useful template for quick and dirty inspection
            #     import matplotlib.pyplot as plt
            #     from pprint import pprint
            #     from mpl_toolkits.axes_grid1 import ImageGrid
            #     from utils import soft_length

            #     print(data["filenames"])
            #     pprint(data["bounds"])
            #     pprint(soft_length(mask_receptacle))

            #     fig = plt.figure()
            #     fig.clear()

            #     grid = ImageGrid(fig, 211, nrows_ncols=(1, 2))

            #     grid[0].imshow(data["images"][0, 0], cmap="gray")
            #     grid[0].contour(data["gt"][0, 1], cmap='jet', alpha=.75, linewidths=2)

            #     grid[1].imshow(data["images"][0, 0], cmap="gray")
            #     grid[1].contour(mask_receptacle[0, 1], cmap='jet', alpha=.75, linewidths=2)
            #     plt.show()

            # Save images
            if savedir:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    predicted_class: Tensor = probs2class(pred_probs)
                    save_images(predicted_class, data["filenames"], savedir, mode, epc)

            # Logging
            big_slice = slice(0, done_img + B)  # Value for current and previous batches

            dsc_dict: Dict
            if few_axis:
                dsc_dict = {**{f"DSC{n}": all_dices[big_slice, n].mean() for n in metric_axis},
                            **({f"3d_DSC{n}": three_d_dices[:done_batch, n].mean() for n in metric_axis}
                                if three_d_dices is not None else {})}
            else:
                dsc_dict = {}

            # dsc_dict = {f"DSC{n}": all_dices[big_slice, n].mean() for n in metric_axis} if few_axis else {}

            hauss_dict = {f"HD{n}": hausdorff_log[big_slice, n].mean() for n in metric_axis} \
                if hausdorff_log is not None and few_axis else {}

            miou_dict = {f"iIoU": iiou_log[big_slice, metric_axis].mean(),
                         f"mIoU": (intersections.sum(dim=0) / (unions.sum(dim=0) + 1e-10)).mean()} \
                if iiou_log is not None and intersections is not None and unions is not None else {}

            if len(metric_axis) > 1:
                mean_dict = {"DSC": all_dices[big_slice, metric_axis].mean()}
                if hausdorff_log:
                    mean_dict["HD"] = hausdorff_log[big_slice, metric_axis].mean()
            else:
                mean_dict = {}

            stat_dict = {**miou_dict, **dsc_dict, **hauss_dict, **mean_dict,
                         "loss": loss_log[:done_batch].mean()}
            nice_dict = {k: f"{v:.3f}" for (k, v) in stat_dict.items()}

            done_img += B
            done_batch += 1
            tq_iter.set_postfix({**nice_dict, "loader": str(i)})
            tq_iter.update(1)
    tq_iter.close()
    print(f"{desc} " + ', '.join(f"{k}={v}" for (k, v) in nice_dict.items()))

    mIoUs: Optional[Tensor]
    if intersections and unions:
        mIoUs = (intersections.sum(dim=0) / (unions.sum(dim=0) + 1e-10))
        assert mIoUs.shape == (K,), mIoUs.shape
    else:
        mIoUs = None

    if not few_axis and False:
        print(f"DSC: {[f'{all_dices[:, n].mean():.3f}' for n in metric_axis]}")
        print(f"iIoU: {[f'{iiou_log[:, n].mean():.3f}' for n in metric_axis]}")
        if mIoUs:
            print(f"mIoU: {[f'{mIoUs[n]:.3f}' for n in metric_axis]}")

    return (loss_log.detach().cpu(),
            all_dices.detach().cpu(),
            hausdorff_log.detach().cpu() if hausdorff_log is not None else None,
            mIoUs.detach().cpu() if mIoUs is not None else None,
            three_d_dices.detach().cpu() if three_d_dices is not None else None,
            number_con.cpu(),
            number_sat_con.cpu())


def do_epoch_cggd_box_prior_box_size_residual_unet(mode: str, net: Any, device: Any, loaders: List[DataLoader], epc: int,
                                                   list_loss_fns: List[List[Callable]], list_loss_weights: List[List[float]], K: int,
                                                   savedir: str = "", optimizer: Any = None,
                                                   metric_axis: List[int] = [1], compute_hausdorff: bool = False, compute_miou: bool = False,
                                                   compute_3d_dice: bool = False,
                                                   temperature: float = 1) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Tensor, Tensor]:
    assert mode in ["train", "val"]

    if mode == "train":
        net.train()
        desc = f">> Training   ({epc})"
    elif mode == "val":
        net.eval()
        desc = f">> Validation ({epc})"

    total_iteration: int = sum(len(loader) for loader in loaders)  # U
    total_images: int = sum(len(loader.dataset) for loader in loaders)  # D
    n_loss: int = max(map(len, list_loss_fns))

    all_dices: Tensor = torch.zeros((total_images, K), dtype=torch.float32, device=device)
    loss_log: Tensor = torch.zeros((total_iteration, n_loss), dtype=torch.float32, device=device)

    iiou_log: Optional[Tensor]
    intersections: Optional[Tensor]
    unions: Optional[Tensor]
    if compute_miou:
        iiou_log = torch.zeros((total_images, K), dtype=torch.float32, device=device)
        intersections = torch.zeros((total_images, K), dtype=torch.float32, device=device)
        unions = torch.zeros((total_images, K), dtype=torch.float32, device=device)
    else:
        iiou_log = None
        intersections = None
        unions = None

    number_con: Tensor = torch.zeros((1, 3), dtype=torch.float32, device=device)
    number_sat_con: Tensor = torch.zeros((1, 3), dtype=torch.float32, device=device)

    three_d_dices: Optional[Tensor]
    if compute_3d_dice:
        three_d_dices = torch.zeros((total_iteration, K), dtype=torch.float32, device=device)
    else:
        three_d_dices = None

    hausdorff_log: Optional[Tensor]
    if compute_hausdorff:
        hausdorff_log = torch.zeros((total_images, K), dtype=torch.float32, device=device)
    else:
        hausdorff_log = None

    few_axis: bool = len(metric_axis) <= 3

    done_img: int = 0
    done_batch: int = 0
    tq_iter = tqdm_(total=total_iteration, desc=desc)
    for i, (loader, loss_fns, loss_weights) in enumerate(zip(loaders, list_loss_fns, list_loss_weights)):
        for data in loader:
            image: Tensor = data["images"].to(device)
            target: Tensor = data["gt"].to(device)
            spacings: Tensor = data["spacings"]  # Keep that one on CPU
            assert not target.requires_grad
            labels: List[Tensor] = [e.to(device) for e in data["labels"]]
            bounds: List[Tensor] = [e.to(device) for e in data["bounds"]]
            box_priors: List[List[Tuple[Tensor, Tensor]]]  # one more level for the batch
            box_priors = [[(m.to(device), b.to(device)) for (m, b) in B] for B in data["box_priors"]]
            assert len(labels) == len(bounds)

            B, C, *_ = image.shape

            samplings: List[List[Tuple[slice]]] = data["samplings"]
            assert len(samplings) == B
            assert len(samplings[0][0]) == len(image[0, 0].shape), (samplings[0][0], image[0, 0].shape)

            probs_receptacle: Tensor = - torch.ones_like(target, dtype=torch.float32)  # -1 for unfilled
            mask_receptacle: Tensor = - torch.ones_like(target, dtype=torch.int32)  # -1 for unfilled

            # Use the sampling coordinates of the first batch item
            assert not (len(samplings[0]) > 1 and B > 1), samplings  # No subsampling if batch size > 1
            loss_sub_log: Tensor = torch.zeros((len(samplings[0]), len(loss_fns)), dtype=torch.float32, device=device)
            for k, sampling in enumerate(samplings[0]):
                img_sampling = [slice(0, B), slice(0, C)] + list(sampling)
                label_sampling = [slice(0, B), slice(0, K)] + list(sampling)
                assert len(img_sampling) == len(image.shape), (img_sampling, image.shape)
                sub_img = image[img_sampling]

                # Reset gradients
                if optimizer:
                    optimizer.zero_grad()

                # Forward
                pred_logits: Tensor = net(sub_img)
                pred_probs: Tensor = F.softmax(temperature * pred_logits, dim=1)
                predicted_mask: Tensor = probs2one_hot(pred_probs.detach())  # Used only for dice computation
                assert not predicted_mask.requires_grad

                probs_receptacle[label_sampling] = pred_probs[...]
                mask_receptacle[label_sampling] = predicted_mask[...]

                assert len(bounds) == len(loss_fns) == len(loss_weights) == len(labels)
                ziped = zip(loss_fns, labels, loss_weights, bounds)
                losses = [w * loss_fn(pred_probs, label[label_sampling], bound, box_priors) for loss_fn, label, w, bound in ziped]
                loss = reduce(add, losses)
                assert loss.shape == (), loss.shape

                ziped = zip(loss_fns, labels, loss_weights, bounds)

                current_loss_fn, current_label, _, current_bound = next(itertools.islice(ziped, 0, None))

                assert simplex(pred_probs)
                B: int = pred_probs.shape[0]
                assert len(box_priors) == B

                number_bounds1 = torch.zeros((), dtype=torch.float32, device=number_con.device)
                number_satisfied_bounds1 = torch.zeros((), dtype=torch.float32, device=number_sat_con.device)

                subdirections = []
                for b in range(B):
                    for l in current_loss_fn.idc:
                        current_masks, current_bounds = box_priors[b][l]

                        current_sizes: Tensor = einsum('wh, nwh->n', pred_probs[b, l], current_masks)

                        assert current_sizes.shape == current_bounds.shape == (current_masks.shape[0],), (
                            current_sizes.shape, current_bounds.shape, current_masks.shape)
                        with torch.no_grad():
                            satisfied_lower_bound1, unsatisfied_lower_bound1, number_satisfied_lower_bound1, number_lower_bound1 = \
                                cggd.check_lower_bound(predictions=current_sizes,
                                                       lower_bound=current_bounds,
                                                       device=device)
                            direction_lower_bound_unsatisfied1 = cggd.compute_direction_lower_bound(unsatisfied_lower_bound=unsatisfied_lower_bound1,
                                                                                                    device=device)

                        init = torch.zeros((), dtype=torch.float32, requires_grad=pred_probs.requires_grad, device=device)
                        subdirections.append(reduce(add, torch.multiply(torch.nn.functional.normalize(direction_lower_bound_unsatisfied1, dim=0), current_sizes), init))
                        init_con = torch.zeros((), dtype=torch.float32, requires_grad=False, device=device)
                        number_bounds1 = torch.add(number_bounds1, reduce(add, (number_lower_bound1, init_con)))
                        number_satisfied_bounds1 = torch.add(number_satisfied_bounds1, reduce(add, (number_satisfied_lower_bound1, init_con)))

                ziped = zip(loss_fns, labels, loss_weights, bounds)

                current_loss_fn, current_label, _, current_bound = next(itertools.islice(ziped, 1, None))
                value2: Tensor = current_loss_fn.__fn__(pred_probs[:, current_loss_fn.idc, ...])

                # Check constraints and compute direction
                with torch.no_grad():
                    lower_bound = current_bound[:, current_loss_fn.idc, :, 0]
                    upper_bound = current_bound[:, current_loss_fn.idc, :, 1]
                    satisfied_lower_bound2, unsatisfied_lower_bound2, number_satisfied_lower_bound2, number_lower_bound2 = \
                        cggd.check_lower_bound(predictions=value2,
                                               lower_bound=lower_bound,
                                               device=device)
                    satisfied_upper_bound2, unsatisfied_upper_bound2, number_satisfied_upper_bound2, number_upper_bound2 = \
                        cggd.check_upper_bound(predictions=value2,
                                               upper_bound=upper_bound,
                                               device=device)

                    direction_lower_bound_unsatisfied2 = cggd.compute_direction_lower_bound(unsatisfied_lower_bound=unsatisfied_lower_bound2,
                                                                                            device=device)
                    direction_upper_bound_unsatisfied2 = cggd.compute_direction_upper_bound(unsatisfied_upper_bound=unsatisfied_upper_bound2,
                                                                                            device=device)

                number_con = torch.add(number_con, torch.reshape(torch.tensor([number_bounds1, number_lower_bound2, number_upper_bound2], dtype=torch.float32, device=number_con.device), [1, 3]))
                number_sat_con = torch.add(number_sat_con, torch.reshape(torch.tensor([number_satisfied_bounds1, number_satisfied_lower_bound2, number_satisfied_upper_bound2], dtype=torch.float32, device=number_sat_con.device), [1, 3]))

                # Backward
                if optimizer:
                    # Apply direction on output with recording of autograph
                    direction_lower_predictions1 = reduce(add, subdirections)
                    direction_lower_predictions2 = torch.sum(torch.multiply(value2, torch.nn.functional.normalize(direction_lower_bound_unsatisfied2, dim=0)))
                    direction_upper_predictions2 = torch.sum(torch.multiply(value2, torch.nn.functional.normalize(direction_upper_bound_unsatisfied2, dim=0)))

                    direction_lower_predictions1.backward(retain_graph=True)

                    list_batch_lower_bound1_grads = []
                    for p in net.parameters():
                        if p.grad is not None:
                            list_batch_lower_bound1_grads.append(p.grad.clone().detach())
                        else:
                            list_batch_lower_bound1_grads.append(None)

                    optimizer.zero_grad()

                    direction_lower_predictions2.backward(retain_graph=True)

                    list_batch_lower_bound2_grads = []
                    for p in net.parameters():
                        if p.grad is not None:
                            list_batch_lower_bound2_grads.append(p.grad.clone().detach())
                        else:
                            list_batch_lower_bound2_grads.append(None)

                    optimizer.zero_grad()

                    direction_upper_predictions2.backward(retain_graph=True)

                    list_batch_upper_bound2_grads = []
                    for p in net.parameters():
                        if p.grad is not None:
                            list_batch_upper_bound2_grads.append(p.grad.clone().detach())
                        else:
                            list_batch_upper_bound2_grads.append(None)

                    optimizer.zero_grad()

                    # combine gradient and directions
                    list_cggd = []
                    for j in range(0, len(list_batch_lower_bound1_grads)):
                        if list_batch_lower_bound1_grads[j] is not None:
                            list_cggd.append(torch.add(list_batch_lower_bound1_grads[j], torch.add(list_batch_lower_bound2_grads[j], list_batch_upper_bound2_grads[j])))
                        else:
                            list_cggd.append(torch.tensor(0, dtype=torch.float32))

                    # store results into trainable parameters
                    counter_parameters = 0
                    for p in net.parameters():
                        if p.grad is not None:
                            p.grad = list_cggd[counter_parameters]
                        counter_parameters += 1

                    # apply CGGD update step
                    optimizer.step()

                # Compute and log metrics
                for j in range(len(loss_fns)):
                    loss_sub_log[k, j] = losses[j].detach()
            reduced_loss_sublog: Tensor = loss_sub_log.sum(dim=0)
            assert reduced_loss_sublog.shape == (len(loss_fns),), (reduced_loss_sublog.shape, len(loss_fns))
            loss_log[done_batch, ...] = reduced_loss_sublog[...]
            del loss_sub_log

            sm_slice = slice(done_img, done_img + B)  # Values only for current batch

            dices: Tensor = dice_coef(mask_receptacle, target)
            assert dices.shape == (B, K), (dices.shape, B, K)
            all_dices[sm_slice, ...] = dices

            if compute_3d_dice:
                three_d_DSC: Tensor = dice_batch(mask_receptacle, target)
                assert three_d_DSC.shape == (K,)

                three_d_dices[done_batch] = three_d_DSC  # type: ignore

            if compute_hausdorff:
                hausdorff_res: Tensor
                try:
                    hausdorff_res = hausdorff(mask_receptacle, target, spacings)
                except RuntimeError:
                    hausdorff_res = torch.zeros((B, K), device=device)
                assert hausdorff_res.shape == (B, K)
                hausdorff_log[sm_slice] = hausdorff_res  # type: ignore
            if compute_miou:
                IoUs: Tensor = iIoU(mask_receptacle, target)
                assert IoUs.shape == (B, K), IoUs.shape
                iiou_log[sm_slice] = IoUs  # type: ignore
                intersections[sm_slice] = inter_sum(mask_receptacle, target)  # type: ignore
                unions[sm_slice] = union_sum(mask_receptacle, target)  # type: ignore

            # if False and target[0, 1].sum() > 0:  # Useful template for quick and dirty inspection
            #     import matplotlib.pyplot as plt
            #     from pprint import pprint
            #     from mpl_toolkits.axes_grid1 import ImageGrid
            #     from utils import soft_length

            #     print(data["filenames"])
            #     pprint(data["bounds"])
            #     pprint(soft_length(mask_receptacle))

            #     fig = plt.figure()
            #     fig.clear()

            #     grid = ImageGrid(fig, 211, nrows_ncols=(1, 2))

            #     grid[0].imshow(data["images"][0, 0], cmap="gray")
            #     grid[0].contour(data["gt"][0, 1], cmap='jet', alpha=.75, linewidths=2)

            #     grid[1].imshow(data["images"][0, 0], cmap="gray")
            #     grid[1].contour(mask_receptacle[0, 1], cmap='jet', alpha=.75, linewidths=2)
            #     plt.show()

            # Save images
            if savedir:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    predicted_class: Tensor = probs2class(pred_probs)
                    save_images(predicted_class, data["filenames"], savedir, mode, epc)

            # Logging
            big_slice = slice(0, done_img + B)  # Value for current and previous batches

            dsc_dict: Dict
            if few_axis:
                dsc_dict = {**{f"DSC{n}": all_dices[big_slice, n].mean() for n in metric_axis},
                            **({f"3d_DSC{n}": three_d_dices[:done_batch, n].mean() for n in metric_axis}
                                if three_d_dices is not None else {})}
            else:
                dsc_dict = {}

            # dsc_dict = {f"DSC{n}": all_dices[big_slice, n].mean() for n in metric_axis} if few_axis else {}

            hauss_dict = {f"HD{n}": hausdorff_log[big_slice, n].mean() for n in metric_axis} \
                if hausdorff_log is not None and few_axis else {}

            miou_dict = {f"iIoU": iiou_log[big_slice, metric_axis].mean(),
                         f"mIoU": (intersections.sum(dim=0) / (unions.sum(dim=0) + 1e-10)).mean()} \
                if iiou_log is not None and intersections is not None and unions is not None else {}

            if len(metric_axis) > 1:
                mean_dict = {"DSC": all_dices[big_slice, metric_axis].mean()}
                if hausdorff_log:
                    mean_dict["HD"] = hausdorff_log[big_slice, metric_axis].mean()
            else:
                mean_dict = {}

            stat_dict = {**miou_dict, **dsc_dict, **hauss_dict, **mean_dict,
                         "loss": loss_log[:done_batch].mean()}
            nice_dict = {k: f"{v:.3f}" for (k, v) in stat_dict.items()}

            done_img += B
            done_batch += 1
            tq_iter.set_postfix({**nice_dict, "loader": str(i)})
            tq_iter.update(1)
    tq_iter.close()
    print(f"{desc} " + ', '.join(f"{k}={v}" for (k, v) in nice_dict.items()))

    mIoUs: Optional[Tensor]
    if intersections and unions:
        mIoUs = (intersections.sum(dim=0) / (unions.sum(dim=0) + 1e-10))
        assert mIoUs.shape == (K,), mIoUs.shape
    else:
        mIoUs = None

    if not few_axis and False:
        print(f"DSC: {[f'{all_dices[:, n].mean():.3f}' for n in metric_axis]}")
        print(f"iIoU: {[f'{iiou_log[:, n].mean():.3f}' for n in metric_axis]}")
        if mIoUs:
            print(f"mIoU: {[f'{mIoUs[n]:.3f}' for n in metric_axis]}")

    return (loss_log.detach().cpu(),
            all_dices.detach().cpu(),
            hausdorff_log.detach().cpu() if hausdorff_log is not None else None,
            mIoUs.detach().cpu() if mIoUs is not None else None,
            three_d_dices.detach().cpu() if three_d_dices is not None else None,
            number_con.cpu(),
            number_sat_con.cpu())


def run(args: argparse.Namespace) -> Dict[str, Tensor]:
    n_class: int = args.n_class
    lr: float = args.l_rate
    savedir: str = args.workdir
    n_epoch: int = args.n_epoch
    val_f: int = args.val_loader_id

    loss_fns: List[List[Callable]]
    loss_weights: List[List[float]]
    net, optimizer, device, loss_fns, loss_weights, scheduler = setup(args, n_class)
    train_loaders: List[DataLoader]
    val_loaders: List[DataLoader]
    train_loaders, val_loaders = get_loaders(args, args.dataset,
                                             args.batch_size, n_class,
                                             args.debug, args.in_memory, args.dimensions, args.use_spacing)

    n_tra: int = sum(len(tr_lo.dataset) for tr_lo in train_loaders)  # Number of images in dataset
    l_tra: int = sum(len(tr_lo) for tr_lo in train_loaders)  # Number of iteration per epc: different if batch_size > 1
    n_val: int = sum(len(vl_lo.dataset) for vl_lo in val_loaders)
    l_val: int = sum(len(vl_lo) for vl_lo in val_loaders)
    n_loss: int = max(map(len, loss_fns))

    best_dice: Tensor = cast(Tensor, torch.zeros(1).type(torch.float32))
    best_sat_ratio: Tensor = cast(Tensor, torch.zeros(1).type(torch.float32))
    best_epoch: int = 0
    reduce_lr_epoch: int = 0
    metrics: Dict[str, Tensor] = {"val_dice": torch.zeros((n_epoch, n_val, n_class)).type(torch.float32),
                                  "val_loss": torch.zeros((n_epoch, l_val, len(loss_fns[val_f]))).type(torch.float32),
                                  "tra_dice": torch.zeros((n_epoch, n_tra, n_class)).type(torch.float32),
                                  "tra_loss": torch.zeros((n_epoch, l_tra, n_loss)).type(torch.float32),
                                  "tra_number_con": torch.zeros((n_epoch, 1, 3)).type(torch.float32),
                                  "tra_number_satisfied_con": torch.zeros((n_epoch, 1, 3)).type(torch.float32),
                                  "val_number_con": torch.zeros((n_epoch, 1, 3)).type(torch.float32),
                                  "val_number_satisfied_con": torch.zeros((n_epoch, 1, 3)).type(torch.float32)}
    if args.compute_hausdorff:
        metrics["val_hausdorff"] = cast(Tensor,
                                        torch.zeros((n_epoch, n_val, n_class)).type(torch.float32))
    if args.compute_miou:
        metrics["val_mIoUs"] = cast(Tensor, torch.zeros((n_epoch, n_class)).type(torch.float32))
        metrics["tra_mIoUs"] = cast(Tensor, torch.zeros((n_epoch, n_class)).type(torch.float32))

    if args.compute_3d_dice:
        metrics["val_3d_dsc"] = cast(Tensor, torch.zeros((n_epoch, l_val, n_class)).type(torch.float32))

    if args.cggd:
        if args.cggd_box_prior and args.cggd_box_size:
            print("\n>>> Starting the training")
            for i in range(n_epoch):
                # Do training and validation loops
                tra_loss, tra_dice, _, tra_mIoUs, _, tra_number_con, tra_number_satisfied_con = \
                    do_epoch_cggd_box_prior_box_size_residual_unet("train", net, device, train_loaders, i,
                                                                   loss_fns, loss_weights, n_class,
                                                                   savedir=savedir if args.save_train else "",
                                                                   optimizer=optimizer,
                                                                   metric_axis=args.metric_axis,
                                                                   compute_miou=args.compute_miou,
                                                                   temperature=args.temperature)
                with torch.no_grad():
                    val_loss, val_dice, val_hausdorff, val_mIoUs, val_3d_dsc, val_number_con, val_number_satisfied_con = \
                        do_epoch_cggd_box_prior_box_size_residual_unet("val", net, device, val_loaders, i,
                                                                       [loss_fns[val_f]],
                                                                       [loss_weights[val_f]],
                                                                       n_class,
                                                                       savedir=savedir,
                                                                       metric_axis=args.metric_axis,
                                                                       compute_hausdorff=args.compute_hausdorff,
                                                                       compute_miou=args.compute_miou,
                                                                       compute_3d_dice=args.compute_3d_dice,
                                                                       temperature=args.temperature)

                # Sort and save the metrics
                for k in metrics:
                    assert metrics[k][i].shape == eval(k).shape, (metrics[k][i].shape, eval(k).shape, k)
                    metrics[k][i] = eval(k)

                for k, e in metrics.items():
                    np.save(Path(savedir, f"{k}.npy"), e.cpu().numpy())

                df = pd.DataFrame({"tra_loss": metrics["tra_loss"].mean(dim=(1, 2)).numpy(),
                                   "val_loss": metrics["val_loss"].mean(dim=(1, 2)).numpy(),
                                   "tra_dice": metrics["tra_dice"][:, :, -1].mean(dim=1).numpy(),
                                   "val_dice": metrics["val_dice"][:, :, -1].mean(dim=1).numpy(),
                                   "tra_sat_ratio": torch.divide(torch.sum(torch.reshape(metrics["tra_number_satisfied_con"], [n_epoch, -1]), dim=1), torch.sum(torch.reshape(metrics["tra_number_con"], [n_epoch, -1]), dim=1)).cpu().numpy(),
                                   "val_sat_ratio": torch.divide(torch.sum(torch.reshape(metrics["val_number_satisfied_con"], [n_epoch, -1]), dim=1), torch.sum(torch.reshape(metrics["val_number_con"], [n_epoch, -1]), dim=1)).cpu().numpy(),
                                   "tra_sat_ratio_bound_1": torch.divide(metrics["tra_number_satisfied_con"][:, 0, 0], metrics["tra_number_con"][:, 0, 0]).cpu().numpy(),
                                   "tra_sat_ratio_bound_2": torch.divide(metrics["tra_number_satisfied_con"][:, 0, 1], metrics["tra_number_con"][:, 0, 1]).cpu().numpy(),
                                   "tra_sat_ratio_bound_3": torch.divide(metrics["tra_number_satisfied_con"][:, 0, 2], metrics["tra_number_con"][:, 0, 2]).cpu().numpy(),
                                   "val_sat_ratio_bound_1": torch.divide(metrics["val_number_satisfied_con"][:, 0, 0], metrics["val_number_con"][:, 0, 0]).cpu().numpy(),
                                   "val_sat_ratio_bound_2": torch.divide(metrics["val_number_satisfied_con"][:, 0, 1], metrics["val_number_con"][:, 0, 1]).cpu().numpy(),
                                   "val_sat_ratio_bound_3": torch.divide(metrics["val_number_satisfied_con"][:, 0, 2], metrics["val_number_con"][:, 0, 2]).cpu().numpy()})
                df.to_csv(Path(savedir, args.csv), float_format="%.4f", index_label="epoch")

                # Save model if better
                #current_dice: Tensor = val_dice[:, args.metric_axis].mean()
                current_dice: Tensor = metrics["val_dice"][:, :, -1].mean(dim=1)
                current_number_con = torch.reshape(torch.sum(torch.reshape(metrics["tra_number_con"], [n_epoch, -1]), dim=1), [n_epoch, 1]).numpy()[i, 0]
                current_number_con_sat = torch.reshape(torch.sum(torch.reshape(metrics["tra_number_satisfied_con"], [n_epoch, -1]), dim=1), [n_epoch, 1]).numpy()[i, 0]
                current_sat_ratio: Tensor = torch.tensor(current_number_con_sat/current_number_con, dtype=torch.float32)

                #if current_dice > best_dice:
                if current_sat_ratio >= best_sat_ratio:
                    if torch.tensor(current_dice.numpy()[i], dtype=torch.float32) > best_dice or current_sat_ratio > best_sat_ratio:
                        best_epoch = i
                        best_dice = current_dice[i]
                    reduce_lr_epoch = i
                    best_sat_ratio = current_sat_ratio
                    if val_hausdorff is not None:
                        best_hausdorff = val_hausdorff[:, args.metric_axis].mean()
                    if val_3d_dsc is not None:
                        best_3d_dsc = val_3d_dsc[:, args.metric_axis].mean()

                    with open(Path(savedir, "best_epoch.txt"), 'w') as f:
                        f.write(str(i))
                    best_folder = Path(savedir, "best_epoch")
                    if best_folder.exists():
                        rmtree(best_folder)
                    copytree(Path(savedir, f"iter{i:03d}"), Path(best_folder))
                    torch.save(net, Path(savedir, "best.pkl"))

                optimizer, loss_fns, loss_weights = scheduler(i, optimizer, loss_fns, loss_weights)

                # if args.schedule and (i > (best_epoch + 20)):
                #if args.schedule and (i % (best_epoch + 20) == 0):  # Yeah, ugly but will clean that later
                if args.schedule and (i % (reduce_lr_epoch + 20) == 0):  # Yeah, ugly but will clean that later
                    for param_group in optimizer.param_groups:
                        lr *= 0.5
                        param_group['lr'] = lr
                        print(f'>> New learning Rate: {lr}')

                if i > 0 and not (i % 5):
                    maybe_hauss = f', Haussdorf: {best_hausdorff:.3f}' if args.compute_hausdorff else ''
                    maybe_3d = f', 3d_DSC: {best_3d_dsc:.3f}' if args.compute_3d_dice else ''
                    print(f">> Best results at epoch {best_epoch}: DSC: {best_dice:.3f}{maybe_hauss}{maybe_3d}")
    else:
        print("\n>>> Starting the training")
        for i in range(n_epoch):
            # Do training and validation loops
            tra_loss, tra_dice, _, tra_mIoUs, _, tra_number_con, tra_number_satisfied_con = \
                do_epoch("train", net, device, train_loaders, i,
                         loss_fns, loss_weights, n_class,
                         savedir=savedir if args.save_train else "",
                         optimizer=optimizer,
                         metric_axis=args.metric_axis,
                         compute_miou=args.compute_miou,
                         temperature=args.temperature)
            with torch.no_grad():
                val_loss, val_dice, val_hausdorff, val_mIoUs, val_3d_dsc, val_number_con, val_number_satisfied_con = \
                    do_epoch("val", net, device, val_loaders, i,
                             [loss_fns[val_f]],
                             [loss_weights[val_f]],
                             n_class,
                             savedir=savedir,
                             metric_axis=args.metric_axis,
                             compute_hausdorff=args.compute_hausdorff,
                             compute_miou=args.compute_miou,
                             compute_3d_dice=args.compute_3d_dice,
                             temperature=args.temperature)

            # Sort and save the metrics
            for k in metrics:
                assert metrics[k][i].shape == eval(k).shape, (metrics[k][i].shape, eval(k).shape, k)
                metrics[k][i] = eval(k)

            for k, e in metrics.items():
                np.save(Path(savedir, f"{k}.npy"), e.cpu().numpy())

            df = pd.DataFrame({"tra_loss": metrics["tra_loss"].mean(dim=(1, 2)).numpy(),
                               "val_loss": metrics["val_loss"].mean(dim=(1, 2)).numpy(),
                               "tra_dice": metrics["tra_dice"][:, :, -1].mean(dim=1).numpy(),
                               "val_dice": metrics["val_dice"][:, :, -1].mean(dim=1).numpy(),
                               "tra_sat_ratio": torch.divide(torch.sum(torch.reshape(metrics["tra_number_satisfied_con"], [n_epoch, -1]), dim=1), torch.sum(torch.reshape(metrics["tra_number_con"], [n_epoch, -1]), dim=1)).cpu().numpy(),
                                   "val_sat_ratio": torch.divide(torch.sum(torch.reshape(metrics["val_number_satisfied_con"], [n_epoch, -1]), dim=1), torch.sum(torch.reshape(metrics["val_number_con"], [n_epoch, -1]), dim=1)).cpu().numpy(),
                               "tra_sat_ratio_bound_1": torch.divide(metrics["tra_number_satisfied_con"][:, 0, 0], metrics["tra_number_con"][:, 0, 0]).cpu().numpy(),
                               "tra_sat_ratio_bound_2": torch.divide(metrics["tra_number_satisfied_con"][:, 0, 1], metrics["tra_number_con"][:, 0, 1]).cpu().numpy(),
                               "tra_sat_ratio_bound_3": torch.divide(metrics["tra_number_satisfied_con"][:, 0, 2], metrics["tra_number_con"][:, 0, 2]).cpu().numpy(),
                               "val_sat_ratio_bound_1": torch.divide(metrics["val_number_satisfied_con"][:, 0, 0], metrics["val_number_con"][:, 0, 0]).cpu().numpy(),
                               "val_sat_ratio_bound_2": torch.divide(metrics["val_number_satisfied_con"][:, 0, 1], metrics["val_number_con"][:, 0, 1]).cpu().numpy(),
                               "val_sat_ratio_bound_3": torch.divide(metrics["val_number_satisfied_con"][:, 0, 2], metrics["val_number_con"][:, 0, 2]).cpu().numpy()})
            df.to_csv(Path(savedir, args.csv), float_format="%.4f", index_label="epoch")

            # Save model if better
            current_dice: Tensor = val_dice[:, args.metric_axis].mean()
            if current_dice > best_dice:
                best_epoch = i
                best_dice = current_dice
                if val_hausdorff is not None:
                    best_hausdorff = val_hausdorff[:, args.metric_axis].mean()
                if val_3d_dsc is not None:
                    best_3d_dsc = val_3d_dsc[:, args.metric_axis].mean()

                with open(Path(savedir, "best_epoch.txt"), 'w') as f:
                    f.write(str(i))
                best_folder = Path(savedir, "best_epoch")
                if best_folder.exists():
                    rmtree(best_folder)
                copytree(Path(savedir, f"iter{i:03d}"), Path(best_folder))
                torch.save(net, Path(savedir, "best.pkl"))

            optimizer, loss_fns, loss_weights = scheduler(i, optimizer, loss_fns, loss_weights)

            # if args.schedule and (i > (best_epoch + 20)):
            if args.schedule and (i % (best_epoch + 20) == 0):  # Yeah, ugly but will clean that later
                for param_group in optimizer.param_groups:
                    lr *= 0.5
                    param_group['lr'] = lr
                    print(f'>> New learning Rate: {lr}')

            if i > 0 and not (i % 5):
                maybe_hauss = f', Haussdorf: {best_hausdorff:.3f}' if args.compute_hausdorff else ''
                maybe_3d = f', 3d_DSC: {best_3d_dsc:.3f}' if args.compute_3d_dice else ''
                print(f">> Best results at epoch {best_epoch}: DSC: {best_dice:.3f}{maybe_hauss}{maybe_3d}")

    # Because displaying the results at the end is actually convenient
    maybe_hauss = f', Haussdorf: {best_hausdorff:.3f}' if args.compute_hausdorff else ''
    maybe_3d = f', 3d_DSC: {best_3d_dsc:.3f}' if args.compute_3d_dice else ''
    print(f">> Best results at epoch {best_epoch}: DSC: {best_dice:.3f}{maybe_hauss}{maybe_3d}")
    for metric in metrics:
        if "val" in metric or "loss" in metric:  # Do not care about training values, nor the loss (keep it simple)
            print(f"\t{metric}: {metrics[metric][best_epoch].mean(dim=0)}")

    return metrics


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--dataset', type=str, required=True)
    # parser.add_argument('--weak_subfolder', type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--losses", type=str, required=True,
                        help="List of list of (loss_name, loss_params, bounds_name, bounds_params, fn, weight)")
    parser.add_argument("--folders", type=str, required=True,
                        help="List of list of (subfolder, transform, is_hot)")
    parser.add_argument("--network", type=str, required=True, help="The network to use")
    parser.add_argument("--n_class", type=int, required=True)
    parser.add_argument("--metric_axis", type=int, nargs='*', required=True, help="Classes to display metrics. \
        Display only the average of everything if empty")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--in_memory", action='store_true')
    parser.add_argument("--schedule", action='store_true')
    parser.add_argument("--use_sgd", action='store_true')
    parser.add_argument("--compute_hausdorff", action='store_true')
    parser.add_argument("--compute_3d_dice", action='store_true')
    parser.add_argument("--compute_miou", action='store_true')
    parser.add_argument("--save_train", action='store_true')
    parser.add_argument("--use_spacing", action='store_true')
    parser.add_argument("--three_d", action='store_true')
    parser.add_argument("--box_prior", action='store_true')
    parser.add_argument("--group", action='store_true', help="Group the patient slices together for validation. \
        Useful to compute the 3d dice, but might destroy the memory for datasets with a lot of slices per patient.")
    parser.add_argument("--group_train", action='store_true', help="Group the patient slices together for training. \
        Useful to compute the 3d dice, but might destroy the memory for datasets with a lot of slices per patient.")

    parser.add_argument('--n_epoch', nargs='?', type=int, default=200,
                        help='# of the epochs')
    parser.add_argument('--l_rate', nargs='?', type=float, default=5e-4,
                        help='Learning Rate')
    parser.add_argument("--grp_regex", type=str, default=None)
    parser.add_argument('--temperature', type=float, default=1, help="Temperature for the softmax")
    parser.add_argument("--scheduler", type=str, default="DummyScheduler")
    parser.add_argument("--scheduler_params", type=str, default="{}")
    parser.add_argument("--modalities", type=int, default=1)
    parser.add_argument("--dimensions", type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--weights", type=str, default='', help="Stored weights to restore")
    parser.add_argument("--training_folders", type=str, nargs="+", default=["train"])
    parser.add_argument("--validation_folder", type=str, default="val")
    parser.add_argument("--val_loader_id", type=int, default=-1, help="""
                        Kinda housefiry at the moment. When we have several train loader (for hybrid training
                        for instance), wants only one validation loader. The way the dataloading creation is
                        written at the moment, it will create several validation loader on the same topfolder (val),
                        but with different folders/bounds ; which will basically duplicate the evaluation.
                        """)
    parser.add_argument("--box_prior_args", type=str, default='')
    parser.add_argument("--gpu_number", type=str, default='2')

    parser.add_argument("--cggd", action='store_true')
    parser.add_argument("--cggd_box_prior", action='store_true')
    parser.add_argument("--cggd_box_size", action='store_true')
    parser.add_argument("--network_seed", type=int, default=5, help="""The pseudo random seed used to initialize the weights of the neural network.""")

    args = parser.parse_args()
    if args.metric_axis == []:
        args.metric_axis = list(range(args.n_class))
    print("\n", args)

    return args


if __name__ == '__main__':
    run(get_args())