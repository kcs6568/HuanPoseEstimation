from random import random
from numpy import dtype, float64
import numpy
import torch
import torch.nn as nn
import numpy as np
import random
from scipy.spatial import distance
import torch.distributed as dist
from torch.autograd import Variable
import mmcv
from collections import OrderedDict

def parse_losses(losses):
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

    return loss, log_vars


a = [torch.rand(5), torch.rand(5)]

print(a)
print(a[0].mean(dim=0))
print(a[0].mean(dim=-1))


# inputs = torch.rand(2, 17, 2)
# zero_idx = torch.randint(
#     low=0,
#     high=16,
#     size=(14,)
# )
# print(zero_idx)

# for i in range(len(inputs)):
#     for idx in sorted(zero_idx):
#         inputs[i][idx] = torch.zeros((2))

# # exit()
# print(inputs)
    

# seq = nn.Sequential(
#     nn.LayerNorm(inputs.size()[1:])
# )
# print(inputs)
# output = seq(inputs)
# print(output)
