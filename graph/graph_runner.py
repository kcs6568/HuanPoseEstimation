import os
import os.path as osp
import numpy as np
import copy
from scipy.spatial import distance
from collections import OrderedDict

import mmcv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist

from brl_graph.graph.iterative_graph import IterativeGraph

def train_model(
    backbone,
    model,
    optimizer,
    criterion,
    dataloader):

    total = 0
    cls_counter = np.zeros((17), dtype=np.int)
    for iter, batch in enumerate(mmcv.track_iter_progress(dataloader)):
    # for iter, batch in enumerate(dataloader):

        losses = torch.tensor(0.)
        optimizer.zero_grad()
        backbone_results = backbone(
            batch['img'],
            batch['targets'],
            batch['masks'],
            batch['joints'],
            return_loss=False,
            return_heatmap=True)

        low_res, high_res = backbone_results[0], backbone_results[1]
        exit()

        ####################
        # data preprocessing
        ####################
        # print("---"*60)
        loc_pred, loc_targets, counter_result = model.module(batch['targets'][1], high_res)
        cls_counter += counter_result
        
        # print(loc_pred)
        # print(loc_targets)
        # exit()

        # print("\n" + "***"*60)
        # print(f'{iter} iteration count: {cls_counter}')
        # print("---"*60 + "\n\n")
        

        for pred, target in zip(loc_pred, loc_targets):
            losses += criterion(pred, target)
        # loss = reg_loss(loc_pred, loc_targets)
        losses.backward()
        optimizer.step()
            

# def train_step(
#     data_batch,
#     optimizer,
#     **kwargs):
    

class GraphRunner(IterativeGraph):
    def __init__(self, backbone_dict, num_joints):
        super().__init__(backbone_dict, num_joints)
        self.reg_loss = nn.L1Loss()
        

    def train_step(self, data_batch, optimizer, **kwargs):
        pred, target = self.forward(data_batch)

        losses = self._get_losses(pred, target)

        loss, log_vars = self._parse_losses(losses)

        output = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        # return pred, target
        return output


    def val_step(self, data_batch):
        results = self.forward(data_batch, train_mode=False)

        outputs = dict(results=results)

        return outputs


    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        # print(losses.items())
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, float):
                log_vars[loss_name] = loss_value
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors or float')

        # if len(losses) > 1:
        loss = sum(_value for _key, _value in log_vars.items()
                if 'loss' in _key)
        log_vars['loss'] = loss
        

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if not isinstance(loss_value, float):
                if dist.is_available() and dist.is_initialized():
                    loss_value = loss_value.data.clone()
                    dist.all_reduce(loss_value.div_(dist.get_world_size()))
                log_vars[loss_name] = loss_value.item()
            else:
                log_vars[loss_name] = loss_value

        # print(loss)
        # print(log_vars)

        return loss, log_vars


    def _get_losses(self, pred, target):
        losses = dict()
        graph_losses = torch.zeros((len(target)), dtype=torch.float32).cuda()

        for idx in range(len(target)):
            loss = self.reg_loss(pred[idx], target[idx]).cuda()
            graph_losses[idx] = loss

        losses['loss'] = graph_losses.mean(dim=0)


        return losses

        



        


        