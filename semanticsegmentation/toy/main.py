#!/usr/bin/env python3.6

import argparse
import warnings
import itertools
import cggd
from pathlib import Path
from functools import reduce
from operator import add, itemgetter
from shutil import copytree, rmtree
from typing import Any, Callable, Dict, List, Tuple

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from networks import weights_init
from dataloader import get_loaders
from utils import map_
from utils import dice_coef, dice_batch, save_images, tqdm_, haussdorf, iIoU
from utils import inter_sum, union_sum
from utils import probs2one_hot, probs2class
from utils import depth


def setup(args, n_class: int) -> Tuple[Any, Any, Any, List[List[Callable]], List[List[float]], Callable]:
    print("\n>>> Setting up")
    cpu: bool = args.cpu or not torch.cuda.is_available()
    device = torch.device("cpu") if cpu else torch.device("cuda:" + args.gpu_number)
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
        net.apply(weights_init)
    net.to(device)

    if args.use_sgd:
        optimizer = torch.optim.SGD(net.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=args.l_rate, betas=(0.9, 0.99), amsgrad=False)

    # print(args.losses)
    list_losses = eval(args.losses)
    if depth(list_losses) == 1:  # For compatibility reasons, avoid changing all the previous configuration files
        list_losses = [list_losses]

    loss_fns: List[List[Callable]] = []
    for i, losses in enumerate(list_losses):
        print(f">> {i}th list of losses: {losses}")
        tmp: List[Callable] = []
        for loss_name, loss_params, _, _, fn, _ in losses:
            loss_class = getattr(__import__('losses'), loss_name)
            tmp.append(loss_class(**loss_params, fn=fn))
        loss_fns.append(tmp)

    loss_weights: List[List[float]] = [map_(itemgetter(5), losses) for losses in list_losses]

    scheduler = getattr(__import__('scheduler'), args.scheduler)(**eval(args.scheduler_params))

    return net, optimizer, device, loss_fns, loss_weights, scheduler


def do_epoch(mode: str, net: Any, device: Any, loaders: List[DataLoader], epc: int,
             list_loss_fns: List[List[Callable]], list_loss_weights: List[List[float]], C: int,
             savedir: str = "", optimizer: Any = None,
             metric_axis: List[int] = [1], compute_haussdorf: bool = False, compute_miou: bool = False,
             temperature: float = 1) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tuple[None, Tensor], Tensor, Tensor]:
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

    all_dices: Tensor = torch.zeros((total_images, C), dtype=torch.float32, device=device)
    batch_dices: Tensor = torch.zeros((total_iteration, C), dtype=torch.float32, device=device)
    loss_log: Tensor = torch.zeros((total_iteration, n_loss), dtype=torch.float32, device=device)
    haussdorf_log: Tensor = torch.zeros((total_images, C), dtype=torch.float32, device=device)
    iiou_log: Tensor = torch.zeros((total_images, C), dtype=torch.float32, device=device)
    intersections: Tensor = torch.zeros((total_images, C), dtype=torch.float32, device=device)
    unions: Tensor = torch.zeros((total_images, C), dtype=torch.float32, device=device)
    number_con: Tensor = torch.zeros((1, 4), dtype=torch.float32, device=device)
    number_sat_con: Tensor = torch.zeros((1, 4), dtype=torch.float32, device=device)

    few_axis: bool = len(metric_axis) <= 3

    done_img: int = 0
    done_batch: int = 0
    tq_iter = tqdm_(total=total_iteration, desc=desc)
    for i, (loader, loss_fns, loss_weights) in enumerate(zip(loaders, list_loss_fns, list_loss_weights)):
        L: int = len(loss_fns)

        for data in loader:
            data[1:] = [e.to(device) for e in data[1:]]  # Move all tensors to device
            filenames, image, target = data[:3]
            assert not target.requires_grad
            labels = data[3:3 + L]
            bounds = data[3 + L:]
            assert len(labels) == len(bounds)

            B = len(image)

            # Reset gradients
            if optimizer:
                optimizer.zero_grad()

            # Forward
            pred_logits: Tensor = net(image)
            pred_probs: Tensor = F.softmax(temperature * pred_logits, dim=1)
            predicted_mask: Tensor = probs2one_hot(pred_probs.detach())  # Used only for dice computation
            assert not predicted_mask.requires_grad

            assert len(bounds) == len(loss_fns) == len(loss_weights) == len(labels)
            ziped = zip(loss_fns, labels, loss_weights, bounds)
            losses = [w * loss_fn(pred_probs, label, bound) for loss_fn, label, w, bound in ziped]
            loss = reduce(add, losses)
            assert loss.shape == (), loss.shape

            # if epc >= 1 and False:
            #     import matplotlib.pyplot as plt
            #     _, axes = plt.subplots(nrows=1, ncols=3)
            #     axes[0].imshow(image[0, 0].cpu().numpy(), cmap='gray')
            #     axes[0].contour(target[0, 1].cpu().numpy(), cmap='rainbow')

            #     pred_np = pred_probs[0, 1].detach().cpu().numpy()
            #     axes[1].imshow(pred_np)

            #     bins = np.linspace(0, 1, 50)
            #     axes[2].hist(pred_np.flatten(), bins)
            #     print(bounds)
            #     print(bounds[2].cpu().numpy())
            #     print(bounds[2][0, 1].cpu().numpy())
            #     print(pred_np.sum())
            #     plt.show()

            # Backward
            if optimizer:
                loss.backward()
                optimizer.step()

            ziped = zip(loss_fns, labels, loss_weights, bounds)

            # it is assumed that first the size constraint is used.
            current_loss_fn, current_label, _, current_bound = next(itertools.islice(ziped, len(loss_fns)-2, None))
            value1: Tensor = current_loss_fn.__fn__(pred_probs[:, current_loss_fn.idc, ...])

            # Check constraints and compute direction
            with torch.no_grad():
                lower_bound = current_bound[:, current_loss_fn.idc, :, 0]
                upper_bound = current_bound[:, current_loss_fn.idc, :, 1]
                satisfied_lower_bound1, unsatisfied_lower_bound1, number_satisfied_lower_bound1, number_lower_bound1 = \
                    cggd.check_lower_bound(predictions=value1,
                                           lower_bound=lower_bound,
                                           device=device)
                satisfied_upper_bound1, unsatisfied_upper_bound1, number_satisfied_upper_bound1, number_upper_bound1 = \
                    cggd.check_upper_bound(predictions=value1,
                                           upper_bound=upper_bound,
                                           device=device)

            ziped = zip(loss_fns, labels, loss_weights, bounds)

            current_loss_fn, current_label, _, current_bound = next(itertools.islice(ziped, len(loss_fns)-1, None))
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

            number_con = torch.add(number_con, torch.reshape(torch.Tensor([number_upper_bound1.to(device), number_lower_bound1.to(device), number_upper_bound2.to(device), number_lower_bound2.to(device)]), (1, 4)).to(device).type(torch.float32))
            number_sat_con = torch.add(number_sat_con, torch.reshape(torch.Tensor([number_satisfied_upper_bound1.to(device), number_satisfied_lower_bound1.to(device), number_satisfied_upper_bound2.to(device), number_satisfied_lower_bound2.to(device)]), (1, 4)).to(device).type(torch.float32))

            # Compute and log metrics
            # loss_log[done_batch] = loss.detach()
            for j in range(len(loss_fns)):
                loss_log[done_batch, j] = losses[j].detach()

            sm_slice = slice(done_img, done_img + B)  # Values only for current batch

            dices: Tensor = dice_coef(predicted_mask, target)
            assert dices.shape == (B, C), (dices.shape, B, C)
            all_dices[sm_slice, ...] = dices

            if B > 1 and mode == "val":
                batch_dice: Tensor = dice_batch(predicted_mask, target)
                assert batch_dice.shape == (C,), (batch_dice.shape, B, C)
                batch_dices[done_batch] = batch_dice

            if compute_haussdorf:
                haussdorf_res: Tensor = haussdorf(predicted_mask, target)
                assert haussdorf_res.shape == (B, C)
                haussdorf_log[sm_slice] = haussdorf_res
            if compute_miou:
                IoUs: Tensor = iIoU(predicted_mask, target)
                assert IoUs.shape == (B, C), IoUs.shape
                iiou_log[sm_slice] = IoUs
                intersections[sm_slice] = inter_sum(predicted_mask, target)
                unions[sm_slice] = union_sum(predicted_mask, target)

            # Save images
            if savedir:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    predicted_class: Tensor = probs2class(pred_probs)
                    save_images(predicted_class, filenames, savedir, mode, epc)

            # Logging
            big_slice = slice(0, done_img + B)  # Value for current and previous batches

            dsc_dict = {f"DSC{n}": all_dices[big_slice, n].mean() for n in metric_axis} if few_axis else {}

            hauss_dict = {f"HD{n}": haussdorf_log[big_slice, n].mean() for n in metric_axis} \
                if compute_haussdorf and few_axis else {}

            batch_dict = {f"bDSC{n}": batch_dices[:done_batch, n].mean() for n in metric_axis} \
                if B > 1 and mode == "val" and few_axis else {}

            miou_dict = {f"iIoU": iiou_log[big_slice, metric_axis].mean(),
                         f"mIoU": (intersections.sum(dim=0) / (unions.sum(dim=0) + 1e-10)).mean()} \
                if compute_miou else {}

            if len(metric_axis) > 1:
                mean_dict = {"DSC": all_dices[big_slice, metric_axis].mean()}
                if compute_haussdorf:
                    mean_dict["HD"] = haussdorf_log[big_slice, metric_axis].mean()
            else:
                mean_dict = {}

            stat_dict = {**miou_dict, **dsc_dict, **hauss_dict, **mean_dict, **batch_dict,
                         "loss": loss_log[:done_batch].mean()}
            nice_dict = {k: f"{v:.3f}" for (k, v) in stat_dict.items()}

            done_img += B
            done_batch += 1
            tq_iter.set_postfix({**nice_dict, "loader": str(i)})
            tq_iter.update(1)
    tq_iter.close()
    print(f"{desc} " + ', '.join(f"{k}={v}" for (k, v) in nice_dict.items()))

    if compute_miou:
        mIoUs: Tensor = (intersections.sum(dim=0) / (unions.sum(dim=0) + 1e-10))
        assert mIoUs.shape == (C,), mIoUs.shape
    else:
        mIoUs = None

    if not few_axis and False:
        print(f"DSC: {[f'{all_dices[:, n].mean():.3f}' for n in metric_axis]}")
        print(f"iIoU: {[f'{iiou_log[:, n].mean():.3f}' for n in metric_axis]}")
        if mIoUs:
            print(f"mIoU: {[f'{mIoUs[n]:.3f}' for n in metric_axis]}")

    return loss_log, all_dices, batch_dices, haussdorf_log, mIoUs, number_con, number_sat_con


# This function is not necessary for the moment if the function above is implemented correctly.
def do_epoch_fs(mode: str, net: Any, device: Any, loaders: List[DataLoader], epc: int,
                list_loss_fns: List[List[Callable]], list_loss_weights: List[List[float]], C: int,
                savedir: str = "", optimizer: Any = None,
                metric_axis: List[int] = [1], compute_haussdorf: bool = False, compute_miou: bool = False,
                temperature: float = 1) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tuple[None, Tensor], Tensor]:
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

    all_dices: Tensor = torch.zeros((total_images, C), dtype=torch.float32, device=device)
    batch_dices: Tensor = torch.zeros((total_iteration, C), dtype=torch.float32, device=device)
    loss_log: Tensor = torch.zeros((total_iteration, n_loss), dtype=torch.float32, device=device)
    haussdorf_log: Tensor = torch.zeros((total_images, C), dtype=torch.float32, device=device)
    iiou_log: Tensor = torch.zeros((total_images, C), dtype=torch.float32, device=device)
    intersections: Tensor = torch.zeros((total_images, C), dtype=torch.float32, device=device)
    unions: Tensor = torch.zeros((total_images, C), dtype=torch.float32, device=device)
    number_con: Tensor = torch.zeros((1, 4), dtype=torch.float32, device=device)
    number_sat_con: Tensor = torch.zeros((1, 4), dtype=torch.float32, device=device)

    few_axis: bool = len(metric_axis) <= 3

    done_img: int = 0
    done_batch: int = 0
    tq_iter = tqdm_(total=total_iteration, desc=desc)
    for i, (loader, loss_fns, loss_weights) in enumerate(zip(loaders, list_loss_fns, list_loss_weights)):
        L: int = len(loss_fns)

        for data in loader:
            data[1:] = [e.to(device) for e in data[1:]]  # Move all tensors to device
            filenames, image, target = data[:3]
            assert not target.requires_grad
            labels = data[3:3 + L]
            bounds = data[3 + L:]
            assert len(labels) == len(bounds)

            B = len(image)

            # Reset gradients
            if optimizer:
                optimizer.zero_grad()

            # Forward
            pred_logits: Tensor = net(image)
            pred_probs: Tensor = F.softmax(temperature * pred_logits, dim=1)
            predicted_mask: Tensor = probs2one_hot(pred_probs.detach())  # Used only for dice computation
            assert not predicted_mask.requires_grad

            assert len(bounds) == len(loss_fns) == len(loss_weights) == len(labels)
            ziped = zip(loss_fns, labels, loss_weights, bounds)
            losses = [w * loss_fn(pred_probs, label, bound) for loss_fn, label, w, bound in ziped]
            loss = reduce(add, losses)
            assert loss.shape == (), loss.shape

            # if epc >= 1 and False:
            #     import matplotlib.pyplot as plt
            #     _, axes = plt.subplots(nrows=1, ncols=3)
            #     axes[0].imshow(image[0, 0].cpu().numpy(), cmap='gray')
            #     axes[0].contour(target[0, 1].cpu().numpy(), cmap='rainbow')

            #     pred_np = pred_probs[0, 1].detach().cpu().numpy()
            #     axes[1].imshow(pred_np)

            #     bins = np.linspace(0, 1, 50)
            #     axes[2].hist(pred_np.flatten(), bins)
            #     print(bounds)
            #     print(bounds[2].cpu().numpy())
            #     print(bounds[2][0, 1].cpu().numpy())
            #     print(pred_np.sum())
            #     plt.show()

            # Backward
            if optimizer:
                loss.backward()
                optimizer.step()

            # Compute and log metrics
            # loss_log[done_batch] = loss.detach()
            for j in range(len(loss_fns)):
                loss_log[done_batch, j] = losses[j].detach()

            sm_slice = slice(done_img, done_img + B)  # Values only for current batch

            dices: Tensor = dice_coef(predicted_mask, target)
            assert dices.shape == (B, C), (dices.shape, B, C)
            all_dices[sm_slice, ...] = dices

            if B > 1 and mode == "val":
                batch_dice: Tensor = dice_batch(predicted_mask, target)
                assert batch_dice.shape == (C,), (batch_dice.shape, B, C)
                batch_dices[done_batch] = batch_dice

            if compute_haussdorf:
                haussdorf_res: Tensor = haussdorf(predicted_mask, target)
                assert haussdorf_res.shape == (B, C)
                haussdorf_log[sm_slice] = haussdorf_res
            if compute_miou:
                IoUs: Tensor = iIoU(predicted_mask, target)
                assert IoUs.shape == (B, C), IoUs.shape
                iiou_log[sm_slice] = IoUs
                intersections[sm_slice] = inter_sum(predicted_mask, target)
                unions[sm_slice] = union_sum(predicted_mask, target)

            # Save images
            if savedir:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    predicted_class: Tensor = probs2class(pred_probs)
                    save_images(predicted_class, filenames, savedir, mode, epc)

            # Logging
            big_slice = slice(0, done_img + B)  # Value for current and previous batches

            dsc_dict = {f"DSC{n}": all_dices[big_slice, n].mean() for n in metric_axis} if few_axis else {}

            hauss_dict = {f"HD{n}": haussdorf_log[big_slice, n].mean() for n in metric_axis} \
                if compute_haussdorf and few_axis else {}

            batch_dict = {f"bDSC{n}": batch_dices[:done_batch, n].mean() for n in metric_axis} \
                if B > 1 and mode == "val" and few_axis else {}

            miou_dict = {f"iIoU": iiou_log[big_slice, metric_axis].mean(),
                         f"mIoU": (intersections.sum(dim=0) / (unions.sum(dim=0) + 1e-10)).mean()} \
                if compute_miou else {}

            if len(metric_axis) > 1:
                mean_dict = {"DSC": all_dices[big_slice, metric_axis].mean()}
                if compute_haussdorf:
                    mean_dict["HD"] = haussdorf_log[big_slice, metric_axis].mean()
            else:
                mean_dict = {}

            stat_dict = {**miou_dict, **dsc_dict, **hauss_dict, **mean_dict, **batch_dict,
                         "loss": loss_log[:done_batch].mean()}
            nice_dict = {k: f"{v:.3f}" for (k, v) in stat_dict.items()}

            done_img += B
            done_batch += 1
            tq_iter.set_postfix({**nice_dict, "loader": str(i)})
            tq_iter.update(1)
    tq_iter.close()
    print(f"{desc} " + ', '.join(f"{k}={v}" for (k, v) in nice_dict.items()))

    if compute_miou:
        mIoUs: Tensor = (intersections.sum(dim=0) / (unions.sum(dim=0) + 1e-10))
        assert mIoUs.shape == (C,), mIoUs.shape
    else:
        mIoUs = None

    if not few_axis and False:
        print(f"DSC: {[f'{all_dices[:, n].mean():.3f}' for n in metric_axis]}")
        print(f"iIoU: {[f'{iiou_log[:, n].mean():.3f}' for n in metric_axis]}")
        if mIoUs:
            print(f"mIoU: {[f'{mIoUs[n]:.3f}' for n in metric_axis]}")

    sat_ratio_individual = torch.divide(number_sat_con, number_con).to(device)
    sat_ratio_combined = torch.divide(torch.sum(number_sat_con), torch.sum(number_con)).to(device)

    sat_ratio = torch.cat((sat_ratio_individual, sat_ratio_combined), dim=1)

    return loss_log, all_dices, batch_dices, haussdorf_log, mIoUs, sat_ratio


def do_epoch_cggd_size_centroid_enet(mode: str, net: Any, device: Any, loaders: List[DataLoader], epc: int,
                                     list_loss_fns: List[List[Callable]], list_loss_weights: List[List[float]], C: int,
                                     savedir: str = "", optimizer: Any = None,
                                     metric_axis: List[int] = [1], compute_haussdorf: bool = False, compute_miou: bool = False,
                                     temperature: float = 1) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tuple[None, Tensor], Tensor, Tensor]:
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

    all_dices: Tensor = torch.zeros((total_images, C), dtype=torch.float32, device=device)
    batch_dices: Tensor = torch.zeros((total_iteration, C), dtype=torch.float32, device=device)
    loss_log: Tensor = torch.zeros((total_iteration, n_loss), dtype=torch.float32, device=device)
    haussdorf_log: Tensor = torch.zeros((total_images, C), dtype=torch.float32, device=device)
    iiou_log: Tensor = torch.zeros((total_images, C), dtype=torch.float32, device=device)
    intersections: Tensor = torch.zeros((total_images, C), dtype=torch.float32, device=device)
    unions: Tensor = torch.zeros((total_images, C), dtype=torch.float32, device=device)
    number_con: Tensor = torch.zeros((1, 4), dtype=torch.float32, device=device)
    number_sat_con: Tensor = torch.zeros((1, 4), dtype=torch.float32, device=device)

    few_axis: bool = len(metric_axis) <= 3

    done_img: int = 0
    done_batch: int = 0
    tq_iter = tqdm_(total=total_iteration, desc=desc)
    for i, (loader, loss_fns, loss_weights) in enumerate(zip(loaders, list_loss_fns, list_loss_weights)):
        L: int = len(loss_fns)

        for data in loader:
            data[1:] = [e.to(device) for e in data[1:]]  # Move all tensors to device
            filenames, image, target = data[:3]
            assert not target.requires_grad
            labels = data[3:3 + L]
            bounds = data[3 + L:]
            assert len(labels) == len(bounds)

            B = len(image)

            # Reset gradients
            if optimizer:
                optimizer.zero_grad()

            # Forward
            pred_logits: Tensor = net(image)
            pred_probs: Tensor = F.softmax(temperature * pred_logits, dim=1)
            predicted_mask: Tensor = probs2one_hot(pred_probs.detach())  # Used only for dice computation
            assert not predicted_mask.requires_grad

            assert len(bounds) == len(loss_fns) == len(loss_weights) == len(labels)
            ziped = zip(loss_fns, labels, loss_weights, bounds)
            losses = [w * loss_fn(pred_probs, label, bound) for loss_fn, label, w, bound in ziped]
            loss = reduce(add, losses)
            assert loss.shape == (), loss.shape

            ziped = zip(loss_fns, labels, loss_weights, bounds)

            # it is assumed that first the size constraint is used.
            current_loss_fn, current_label, _, current_bound = next(itertools.islice(ziped, 0, None))
            value1: Tensor = current_loss_fn.__fn__(pred_probs[:, current_loss_fn.idc, ...])

            # Check constraints and compute direction
            with torch.no_grad():
                lower_bound = current_bound[:, current_loss_fn.idc, :, 0]
                upper_bound = current_bound[:, current_loss_fn.idc, :, 1]
                satisfied_lower_bound1, unsatisfied_lower_bound1, number_satisfied_lower_bound1, number_lower_bound1 = \
                    cggd.check_lower_bound(predictions=value1,
                                           lower_bound=lower_bound,
                                           device=device)
                satisfied_upper_bound1, unsatisfied_upper_bound1, number_satisfied_upper_bound1, number_upper_bound1 = \
                    cggd.check_upper_bound(predictions=value1,
                                           upper_bound=upper_bound,
                                           device=device)

                direction_lower_bound_unsatisfied1 = cggd.compute_direction_lower_bound(unsatisfied_lower_bound=unsatisfied_lower_bound1, device=device)
                direction_upper_bound_unsatisfied1 = cggd.compute_direction_upper_bound(unsatisfied_upper_bound=unsatisfied_upper_bound1, device=device)

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

                direction_lower_bound_unsatisfied2 = cggd.compute_direction_lower_bound(unsatisfied_lower_bound=unsatisfied_lower_bound2, device=device)
                direction_upper_bound_unsatisfied2 = cggd.compute_direction_upper_bound(unsatisfied_upper_bound=unsatisfied_upper_bound2, device=device)

            number_con = torch.add(number_con, torch.reshape(
                torch.Tensor([number_upper_bound1.to(device), number_lower_bound1.to(device), number_upper_bound2.to(device), number_lower_bound2.to(device)]), (1, 4)).to(device).type(
                torch.float32))
            number_sat_con = torch.add(number_sat_con, torch.reshape(
                torch.Tensor([number_satisfied_upper_bound1.to(device), number_satisfied_lower_bound1.to(device), number_satisfied_upper_bound2.to(device), number_satisfied_lower_bound2.to(device)]),
                (1, 4)).to(device).type(torch.float32))
            # Backward
            if optimizer:
                # Apply direction on output with recording of autograph
                direction_lower_predictions1 = torch.sum(torch.multiply(value1, torch.nn.functional.normalize(direction_lower_bound_unsatisfied1, dim=0)))
                direction_upper_predictions1 = torch.sum(torch.multiply(value1, torch.nn.functional.normalize(direction_upper_bound_unsatisfied1, dim=0)))

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

                direction_upper_predictions1.backward(retain_graph=True)

                list_batch_upper_bound1_grads = []
                for p in net.parameters():
                    if p.grad is not None:
                        list_batch_upper_bound1_grads.append(p.grad.clone().detach())
                    else:
                        list_batch_upper_bound1_grads.append(None)

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

                ## normalize and rescale directions of constraints
                #with torch.no_grad():
                #    # normalize (with respect to L2-norm) the direction of size constraint (lower bound)
                #    list_batch_lower_bound1_grads_norm = cggd.normalize_direction_enet(directions=list_batch_lower_bound1_grads, device=device)
                #
                #    # normalize (with respect to L2-norm) the direction of size constraint (upper bound)
                #    list_batch_upper_bound1_grads_norm = cggd.normalize_direction_enet(directions=list_batch_upper_bound1_grads, device=device)
                #
                #    # normalize (with respect to L2-norm) the direction of centroid constraint (lower bound)
                #    list_batch_lower_bound2_grads_norm = cggd.normalize_direction_enet(directions=list_batch_lower_bound2_grads, device=device)
                #
                #    # normalize (with respect to L2-norm) the direction of centroid constraint (upper bound)
                #    list_batch_upper_bound2_grads_norm = cggd.normalize_direction_enet(directions=list_batch_upper_bound2_grads, device=device)

                # combine gradient and directions
                list_cggd = []
                #for j in range(0, len(list_batch_lower_bound1_grads_norm)):
                #    if list_batch_lower_bound1_grads_norm[j] is not None:
                #        list_cggd.append(torch.add(list_batch_lower_bound1_grads_norm[j],
                #                                   torch.add(list_batch_upper_bound1_grads_norm[j],
                #                                             torch.add(list_batch_lower_bound2_grads_norm[j],
                #                                                       list_batch_upper_bound2_grads_norm[j]))))
                #    else:
                #        list_cggd.append(None)
                for j in range(0, len(list_batch_lower_bound1_grads)):
                    if list_batch_lower_bound1_grads[j] is not None:
                        list_cggd.append(torch.add(list_batch_lower_bound1_grads[j],
                                                   torch.add(list_batch_upper_bound1_grads[j],
                                                             torch.add(list_batch_lower_bound2_grads[j],
                                                                       list_batch_upper_bound2_grads[j]))))
                    else:
                        list_cggd.append(None)

                # store results into trainable parameters
                counter_parameters = 0
                for p in net.parameters():
                    if p.grad is not None:
                        p.grad = list_cggd[counter_parameters]
                    counter_parameters += 1

                # apply CGGD update step
                optimizer.step()

            # Compute and log metrics
            # loss_log[done_batch] = loss.detach()
            for j in range(len(loss_fns)):
                loss_log[done_batch, j] = losses[j].detach()

            sm_slice = slice(done_img, done_img + B)  # Values only for current batch

            dices: Tensor = dice_coef(predicted_mask, target)
            assert dices.shape == (B, C), (dices.shape, B, C)
            all_dices[sm_slice, ...] = dices

            if B > 1 and mode == "val":
                batch_dice: Tensor = dice_batch(predicted_mask, target)
                assert batch_dice.shape == (C,), (batch_dice.shape, B, C)
                batch_dices[done_batch] = batch_dice

            if compute_haussdorf:
                haussdorf_res: Tensor = haussdorf(predicted_mask, target)
                assert haussdorf_res.shape == (B, C)
                haussdorf_log[sm_slice] = haussdorf_res
            if compute_miou:
                IoUs: Tensor = iIoU(predicted_mask, target)
                assert IoUs.shape == (B, C), IoUs.shape
                iiou_log[sm_slice] = IoUs
                intersections[sm_slice] = inter_sum(predicted_mask, target)
                unions[sm_slice] = union_sum(predicted_mask, target)

            # Save images
            if savedir:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    predicted_class: Tensor = probs2class(pred_probs)
                    save_images(predicted_class, filenames, savedir, mode, epc)

            # Logging
            big_slice = slice(0, done_img + B)  # Value for current and previous batches

            dsc_dict = {f"DSC{n}": all_dices[big_slice, n].mean() for n in metric_axis} if few_axis else {}

            hauss_dict = {f"HD{n}": haussdorf_log[big_slice, n].mean() for n in metric_axis} \
                if compute_haussdorf and few_axis else {}

            batch_dict = {f"bDSC{n}": batch_dices[:done_batch, n].mean() for n in metric_axis} \
                if B > 1 and mode == "val" and few_axis else {}

            miou_dict = {f"iIoU": iiou_log[big_slice, metric_axis].mean(),
                         f"mIoU": (intersections.sum(dim=0) / (unions.sum(dim=0) + 1e-10)).mean()} \
                if compute_miou else {}

            if len(metric_axis) > 1:
                mean_dict = {"DSC": all_dices[big_slice, metric_axis].mean()}
                if compute_haussdorf:
                    mean_dict["HD"] = haussdorf_log[big_slice, metric_axis].mean()
            else:
                mean_dict = {}

            stat_dict = {**miou_dict, **dsc_dict, **hauss_dict, **mean_dict, **batch_dict,
                         "loss": loss_log[:done_batch].mean()}
            nice_dict = {k: f"{v:.3f}" for (k, v) in stat_dict.items()}

            done_img += B
            done_batch += 1
            tq_iter.set_postfix({**nice_dict, "loader": str(i)})
            tq_iter.update(1)
    tq_iter.close()
    print(f"{desc} " + ', '.join(f"{k}={v}" for (k, v) in nice_dict.items()))

    if compute_miou:
        mIoUs: Tensor = (intersections.sum(dim=0) / (unions.sum(dim=0) + 1e-10))
        assert mIoUs.shape == (C,), mIoUs.shape
    else:
        mIoUs = None

    if not few_axis and False:
        print(f"DSC: {[f'{all_dices[:, n].mean():.3f}' for n in metric_axis]}")
        print(f"iIoU: {[f'{iiou_log[:, n].mean():.3f}' for n in metric_axis]}")
        if mIoUs:
            print(f"mIoU: {[f'{mIoUs[n]:.3f}' for n in metric_axis]}")

    return loss_log, all_dices, batch_dices, haussdorf_log, mIoUs, number_con, number_sat_con


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
                                             args.debug, args.in_memory)

    n_tra: int = sum(len(tr_lo.dataset) for tr_lo in train_loaders)  # Number of images in dataset
    l_tra: int = sum(len(tr_lo) for tr_lo in train_loaders)  # Number of iteration per epc: different if batch_size > 1
    n_val: int = sum(len(vl_lo.dataset) for vl_lo in val_loaders)
    l_val: int = sum(len(vl_lo) for vl_lo in val_loaders)
    n_loss: int = max(map(len, loss_fns))

    best_dice: Tensor = torch.zeros(1).to(device).type(torch.float32)
    best_epoch: int = 0
    #metrics = {"val_dice": torch.zeros((n_epoch, n_val, n_class), device=device).type(torch.float32),
    #           "val_batch_dice": torch.zeros((n_epoch, l_val, n_class), device=device).type(torch.float32),
    #           "val_loss": torch.zeros((n_epoch, l_val, len(loss_fns[val_f])), device=device).type(torch.float32),
    #           "tra_dice": torch.zeros((n_epoch, n_tra, n_class), device=device).type(torch.float32),
    #           "tra_batch_dice": torch.zeros((n_epoch, l_tra, n_class), device=device).type(torch.float32),
    #           "tra_loss": torch.zeros((n_epoch, l_tra, n_loss), device=device).type(torch.float32)}
    metrics = {"val_dice": torch.zeros((n_epoch, n_val, n_class), device=device).type(torch.float32),
               "val_batch_dice": torch.zeros((n_epoch, l_val, n_class), device=device).type(torch.float32),
               "val_loss": torch.zeros((n_epoch, l_val, len(loss_fns[val_f])), device=device).type(torch.float32),
               "tra_dice": torch.zeros((n_epoch, n_tra, n_class), device=device).type(torch.float32),
               "tra_batch_dice": torch.zeros((n_epoch, l_tra, n_class), device=device).type(torch.float32),
               "tra_loss": torch.zeros((n_epoch, l_tra, n_loss), device=device).type(torch.float32),
               "tra_number_con": torch.zeros((n_epoch, 1, 4), device=device).type(torch.float32),
               "tra_sat_number_con": torch.zeros((n_epoch, 1, 4), device=device).type(torch.float32),
               "val_number_con": torch.zeros((n_epoch, 1, 4), device=device).type(torch.float32),
               "val_sat_number_con": torch.zeros((n_epoch, 1, 4), device=device).type(torch.float32)}
    if args.compute_haussdorf:
        metrics["val_haussdorf"] = torch.zeros((n_epoch, n_val, n_class), device=device).type(torch.float32)
    if args.compute_miou:
        metrics["val_mIoUs"] = torch.zeros((n_epoch, n_class), device=device).type(torch.float32)
        metrics["tra_mIoUs"] = torch.zeros((n_epoch, n_class), device=device).type(torch.float32)

    if args.cggd:
        # train with CGGD
        if args.size_con and args.centroid_con:
            # consider both a size constraint and centroid constraint for CGGD
            if args.network == 'ENet':
                # the neural network used is of type ENet.
                print('\n>>> Starting the training')
                for i in range(n_epoch):
                    # Do training and validation loops
                    tra_loss, tra_dice, tra_batch_dice, _, tra_mIoUs, tra_number_con, tra_sat_number_con = \
                        do_epoch_cggd_size_centroid_enet("train", net, device, train_loaders, i,
                                                         loss_fns, loss_weights, n_class,
                                                         savedir=savedir if args.save_train else "",
                                                         optimizer=optimizer,
                                                         metric_axis=args.metric_axis,
                                                         compute_miou=args.compute_miou,
                                                         temperature=args.temperature)
                    with torch.no_grad():
                        val_loss, val_dice, val_batch_dice, val_haussdorf, val_mIoUs, val_number_con, val_sat_number_con = \
                            do_epoch_cggd_size_centroid_enet("val", net, device, val_loaders, i,
                                                             [loss_fns[val_f]],
                                                             [loss_weights[val_f]],
                                                             n_class,
                                                             savedir=savedir,
                                                             metric_axis=args.metric_axis,
                                                             compute_haussdorf=args.compute_haussdorf,
                                                             compute_miou=args.compute_miou,
                                                             temperature=args.temperature)

                    # Sort and save the metrics
                    for k in metrics:
                        assert metrics[k][i].shape == eval(k).shape, (metrics[k][i].shape, eval(k).shape, k)
                        metrics[k][i] = eval(k)

                    for k, e in metrics.items():
                        np.save(Path(savedir, f"{k}.npy"), e.cpu().numpy())

                    df = pd.DataFrame({"tra_loss": metrics["tra_loss"].mean(dim=(1, 2)).cpu().numpy(),
                                       "val_loss": metrics["val_loss"].mean(dim=(1, 2)).cpu().numpy(),
                                       "tra_dice": metrics["tra_dice"][:, :, -1].mean(dim=1).cpu().numpy(),
                                       "val_dice": metrics["val_dice"][:, :, -1].mean(dim=1).cpu().numpy(),
                                       "tra_batch_dice": metrics["tra_batch_dice"][:, :, -1].mean(dim=1).cpu().numpy(),
                                       "val_batch_dice": metrics["val_batch_dice"][:, :, -1].mean(dim=1).cpu().numpy(),
                                       "tra_sat_ratio": torch.divide(torch.sum(metrics["tra_sat_number_con"], dim=(1, 2)), torch.sum(metrics["tra_number_con"], dim=(1, 2))).cpu().numpy(),
                                       "val_sat_ratio": torch.divide(torch.sum(metrics["val_sat_number_con"], dim=(1, 2)), torch.sum(metrics["val_number_con"], dim=(1, 2))).cpu().numpy(),
                                       "tra_sat_ratio_bound_1": torch.divide(metrics["tra_sat_number_con"][:, 0, 0], metrics["tra_number_con"][:, 0, 0]).cpu().numpy(),
                                       "tra_sat_ratio_bound_2": torch.divide(metrics["tra_sat_number_con"][:, 0, 1], metrics["tra_number_con"][:, 0, 1]).cpu().numpy(),
                                       "tra_sat_ratio_bound_3": torch.divide(metrics["tra_sat_number_con"][:, 0, 2], metrics["tra_number_con"][:, 0, 2]).cpu().numpy(),
                                       "tra_sat_ratio_bound_4": torch.divide(metrics["tra_sat_number_con"][:, 0, 3], metrics["tra_number_con"][:, 0, 3]).cpu().numpy(),
                                       "val_sat_ratio_bound_1": torch.divide(metrics["val_sat_number_con"][:, 0, 0], metrics["val_number_con"][:, 0, 0]).cpu().numpy(),
                                       "val_sat_ratio_bound_2": torch.divide(metrics["val_sat_number_con"][:, 0, 1], metrics["val_number_con"][:, 0, 1]).cpu().numpy(),
                                       "val_sat_ratio_bound_3": torch.divide(metrics["val_sat_number_con"][:, 0, 2], metrics["val_number_con"][:, 0, 2]).cpu().numpy(),
                                       "val_sat_ratio_bound_4": torch.divide(metrics["val_sat_number_con"][:, 0, 3], metrics["val_number_con"][:, 0, 3]).cpu().numpy()})
                    df.to_csv(Path(savedir, args.csv), float_format="%.4f", index_label="epoch")

                    # Save model if better
                    current_dice: Tensor = val_dice[:, args.metric_axis].mean()
                    if current_dice > best_dice:
                        best_epoch = i
                        best_dice = current_dice
                        if args.compute_haussdorf:
                            best_haussdorf = val_haussdorf[:, args.metric_axis].mean()

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
                        maybe_hauss = f', Haussdorf: {best_haussdorf:.3f}' if args.compute_haussdorf else ''
                        print(f">> Best results at epoch {best_epoch}: DSC: {best_dice:.3f}{maybe_hauss}")
        else:
            print('''Training with CGGD was selected but no constraint was satisfied. Please make sure that the boolean variables that indicate which
                  constraints are used are set correctly.''')
            sys.exit()
    else:
        # train with method that is not CGGD
        print("\n>>> Starting the training")
        for i in range(n_epoch):
            # Do training and validation loops
            tra_loss, tra_dice, tra_batch_dice, _, tra_mIoUs, tra_number_con, tra_sat_number_con = \
                do_epoch("train", net, device, train_loaders, i,
                         loss_fns, loss_weights, n_class,
                         savedir=savedir if args.save_train else "",
                         optimizer=optimizer,
                         metric_axis=args.metric_axis,
                         compute_miou=args.compute_miou,
                         temperature=args.temperature)
            with torch.no_grad():
                val_loss, val_dice, val_batch_dice, val_haussdorf, val_mIoUs, val_number_con, val_sat_number_con = \
                    do_epoch("val", net, device, val_loaders, i,
                             [loss_fns[val_f]],
                             [loss_weights[val_f]],
                             n_class,
                             savedir=savedir,
                             metric_axis=args.metric_axis,
                             compute_haussdorf=args.compute_haussdorf,
                             compute_miou=args.compute_miou,
                             temperature=args.temperature)

            # Sort and save the metrics
            for k in metrics:
                assert metrics[k][i].shape == eval(k).shape, (metrics[k][i].shape, eval(k).shape, k)
                metrics[k][i] = eval(k)

            for k, e in metrics.items():
                np.save(Path(savedir, f"{k}.npy"), e.cpu().numpy())

            df = pd.DataFrame({"tra_loss": metrics["tra_loss"].mean(dim=(1, 2)).cpu().numpy(),
                               "val_loss": metrics["val_loss"].mean(dim=(1, 2)).cpu().numpy(),
                               "tra_dice": metrics["tra_dice"][:, :, -1].mean(dim=1).cpu().numpy(),
                               "val_dice": metrics["val_dice"][:, :, -1].mean(dim=1).cpu().numpy(),
                               "tra_batch_dice": metrics["tra_batch_dice"][:, :, -1].mean(dim=1).cpu().numpy(),
                               "val_batch_dice": metrics["val_batch_dice"][:, :, -1].mean(dim=1).cpu().numpy(),
                               "tra_sat_ratio": torch.divide(torch.sum(metrics["tra_sat_number_con"], dim=(1, 2)), torch.sum(metrics["tra_number_con"], dim=(1, 2))).cpu().numpy(),
                               "val_sat_ratio": torch.divide(torch.sum(metrics["val_sat_number_con"], dim=(1, 2)), torch.sum(metrics["val_number_con"], dim=(1, 2))).cpu().numpy(),
                               "tra_sat_ratio_bound_1": torch.divide(metrics["tra_sat_number_con"][:, 0, 0], metrics["tra_number_con"][:, 0, 0]).cpu().numpy(),
                               "tra_sat_ratio_bound_2": torch.divide(metrics["tra_sat_number_con"][:, 0, 1], metrics["tra_number_con"][:, 0, 1]).cpu().numpy(),
                               "tra_sat_ratio_bound_3": torch.divide(metrics["tra_sat_number_con"][:, 0, 2], metrics["tra_number_con"][:, 0, 2]).cpu().numpy(),
                               "tra_sat_ratio_bound_4": torch.divide(metrics["tra_sat_number_con"][:, 0, 3], metrics["tra_number_con"][:, 0, 3]).cpu().numpy(),
                               "val_sat_ratio_bound_1": torch.divide(metrics["val_sat_number_con"][:, 0, 0], metrics["val_number_con"][:, 0, 0]).cpu().numpy(),
                               "val_sat_ratio_bound_2": torch.divide(metrics["val_sat_number_con"][:, 0, 1], metrics["val_number_con"][:, 0, 1]).cpu().numpy(),
                               "val_sat_ratio_bound_3": torch.divide(metrics["val_sat_number_con"][:, 0, 2], metrics["val_number_con"][:, 0, 2]).cpu().numpy(),
                               "val_sat_ratio_bound_4": torch.divide(metrics["val_sat_number_con"][:, 0, 3], metrics["val_number_con"][:, 0, 3]).cpu().numpy()})
            df.to_csv(Path(savedir, args.csv), float_format="%.4f", index_label="epoch")

            # Save model if better
            current_dice: Tensor = val_dice[:, args.metric_axis].mean()
            if current_dice > best_dice:
                best_epoch = i
                best_dice = current_dice
                if args.compute_haussdorf:
                    best_haussdorf = val_haussdorf[:, args.metric_axis].mean()

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
                maybe_hauss = f', Haussdorf: {best_haussdorf:.3f}' if args.compute_haussdorf else ''
                print(f">> Best results at epoch {best_epoch}: DSC: {best_dice:.3f}{maybe_hauss}")

    # Because displaying the results at the end is actually convenient
    maybe_hauss = f', Haussdorf: {best_haussdorf:.3f}' if args.compute_haussdorf else ''
    print(f">> Best results at epoch {best_epoch}: DSC: {best_dice:.3f}{maybe_hauss}")
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
    parser.add_argument("--grp_regex", type=str, required=True)
    parser.add_argument("--n_class", type=int, required=True)
    parser.add_argument("--metric_axis", type=int, nargs='*', required=True, help="Classes to display metrics. \
        Display only the average of everything if empty")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--in_memory", action='store_true')
    parser.add_argument("--schedule", action='store_true')
    parser.add_argument("--use_sgd", action='store_true')
    parser.add_argument("--compute_haussdorf", action='store_true')
    parser.add_argument("--compute_miou", action='store_true')
    parser.add_argument("--save_train", action='store_true')
    parser.add_argument("--group", action='store_true', help="Group the patient slices together for validation. \
        Useful to compute the 3d dice, but might destroy the memory for datasets with a lot of slices per patient.")
    parser.add_argument("--group_train", action='store_true', help="Group the patient slices together for training. \
        Useful to compute the 3d dice, but might destroy the memory for datasets with a lot of slices per patient.")

    parser.add_argument('--n_epoch', nargs='?', type=int, default=200,
                        help='# of the epochs')
    parser.add_argument('--l_rate', nargs='?', type=float, default=5e-4,
                        help='Learning Rate')
    parser.add_argument('--temperature', type=float, default=1, help="Temperature for the softmax")
    parser.add_argument("--scheduler", type=str, default="DummyScheduler")
    parser.add_argument("--scheduler_params", type=str, default="{}")
    parser.add_argument("--modalities", type=int, default=1)
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

    # the following arguments are added
    parser.add_argument('--cggd', action='store_true', help='''Boolean indicating if the CGGD method should be used to train with constraints. 
                                                                    Default value is False indicating that a different method should be used.''')
    parser.add_argument('--size_con', action='store_true', help='''Boolean indicating if the size constraint is used for CGGD. Default value is 
                                                                        False indicating that this constraint is not considered.''')
    parser.add_argument('--centroid_con', action='store_true', help='''Boolean indicating if the centroid constraint is used for CGGD. Default 
                                                                            value is False indicating that this constraint is not considered.''')
    parser.add_argument("--network_seed", type=int, default=5, help="""The pseudo random seed used to initialize the weights of the neural network.""")
    parser.add_argument("--gpu_number", type=str, default='3')

    args = parser.parse_args()
    if args.metric_axis == []:
        args.metric_axis = list(range(args.n_class))
    print("\n", args)

    return args


if __name__ == '__main__':
    run(get_args())
