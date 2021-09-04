import math

import torch
import torch.nn as nn

from torch.nn.parameter import Parameter


# class GraphConvolution(nn.Module):
#     def __init__(self, in_channels, out_channels, bias=False):
#         super(GraphConvolution, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.weight = Parameter(torch.FloatTensor(in_channels, out_channels))
#         if bias:
#             self.bias = Parameter(torch.FloatTensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()

    
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)


#     def forward(self, input, adj):
#         support = torch.mm(input, self.weight)
#         output = torch.spmm(adj, support)

#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output


class Layers(nn.Module):
    def __init__(
        self,
        in_channels,
        hid_channels,
        out_channels,
        dropout=0.2,
        bias=False):
        super(Layers, self).__init__()
        self.linear1 = nn.Linear(in_channels, hid_channels)
        self.linear2 = nn.Linear(hid_channels, out_channels)
        # self.dropout = 

    def forward(self, input, adj):
        pass
