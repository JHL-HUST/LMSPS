import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, ChebConv, GraphConv




class Dy(nn.Module):
    def __init__(self, channels, k=1, reduction=4):
        super(Dy, self).__init__()
        self.channels = channels

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        # self.fc2 = nn.Linear(channels // reduction, 2*k)
        self.fc2 = nn.Linear(channels // reduction, k)
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()
        # import code
        # code.interact(local=locals())

    def reset_parameters(self):
        # for layer in self.layers:
        #     if isinstance(layer, Conv1d1x1):
        #         layer.reset_parameters()
        # for k, v in self.feat_gat_layers.items():
        #     v.reset_parameters()

        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.fc1.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc2.weight, gain=gain)
        # nn.init.zeros_(self.layer_final.bias)
        # if self.residual:
        #     nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        # for layer in self.lr_output:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.xavier_uniform_(layer.weight, gain=gain)
        #         if layer.bias is not None:
        #             nn.init.zeros_(layer.bias)

    def get_coefs(self, x):
        theta = x #torch.mean(x, axis=-1)

        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = self.sigmoid(theta)
        return theta

    def forward(self, x):
        x = F.dropout(x, p=0.5, training=self.training)
        alpha = self.get_coefs(x)
        return alpha





class DyReLU(nn.Module):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2*k)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.]*(2*k - 1)).float())

    def get_relu_coefs(self, x):
        theta = torch.mean(x, axis=-1)
        if self.conv_type == '2d':
            theta = torch.mean(theta, axis=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x):
        raise NotImplementedError





class DyReLUA(DyReLU):
    def __init__(self, channels, reduction=1, k=4, conv_type='2d'):
        super(DyReLUA, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        # import code
        # code.interact(local=locals())

        relu_coefs = theta.view(-1, 2*self.k) * self.lambdas + self.init_v
        # BxCxL -> LxCxBx1
        x_perm = x.transpose(0, -1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :self.k] + relu_coefs[:, self.k:]
        # LxCxBx2 -> BxCxL
        result = torch.max(output, dim=-1)[0].transpose(0, -1)

        return result


class DyReLUB(DyReLU):
    def __init__(self, channels, reduction=1, k=2, conv_type='2d'):
        super(DyReLUB, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        # import code
        # code.interact(local=locals())

        relu_coefs = theta.view(-1, 2*self.k) * self.lambdas + self.init_v
        # BxCxL -> LxCxBx1
        x_perm = x.transpose(0, -1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :self.k] + relu_coefs[:, self.k:]
        # LxCxBx2 -> BxCxL
        result = torch.max(output, dim=-1)[0].transpose(0, -1)

        return result



# class DyReLUB(DyReLU):
#     def __init__(self, channels, reduction=1, k=2, conv_type='2d'):
#         super(DyReLUB, self).__init__(channels, reduction, k, conv_type)
#         self.fc2 = nn.Linear(channels // reduction, 2*k*channels)
#
#     def forward(self, x):
#         assert x.shape[1] == self.channels
#
#         # import code
#         # code.interact(local=locals())
#
#         theta = self.get_relu_coefs(x)
#
#         relu_coefs = theta.view(-1, self.channels, 2*self.k) * self.lambdas + self.init_v
#         # BxCxHxW -> HxWxBxCx1
#         x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
#         output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
#         # HxWxBxCx2 -> BxCxHxW
#         result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)
#
#         return result



# class DyReLU(nn.Module):
#     def __init__(self, channels, reduction=4, k=2, conv_type='1d'):
#         super(DyReLU, self).__init__()
#         self.channels = channels
#         self.k = k
#         self.conv_type = conv_type
#         assert self.conv_type in ['1d', '2d']
#
#         self.fc1 = nn.Linear(channels, channels // reduction)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(channels // reduction, 2*k)
#         self.sigmoid = nn.Sigmoid()
#
#         self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
#         self.register_buffer('init_v', torch.Tensor([1.] + [0.]*(2*k - 1)).float())
#
#     def get_relu_coefs(self, x):
#         theta = torch.mean(x, axis=0)
#         import code
#         code.interact(local=locals())
#         theta = self.fc1(theta)
#         theta = self.relu(theta)
#         theta = self.fc2(theta)
#         theta = 2 * self.sigmoid(theta) - 1
#         # import code
#         # code.interact(local=locals())
#         return theta
#
#     def forward(self, x):
#         raise NotImplementedError

#class DyReLUB(DyReLU):
#     def __init__(self, channels, reduction=4, k=2, conv_type='1d'):
#         super(DyReLUB, self).__init__(channels, reduction, k, conv_type)
#         self.fc2 = nn.Linear(channels // reduction, 2*k*channels)
#
#     def forward(self, x):
#         assert x.shape[1] == self.channels
#         theta = self.get_relu_coefs(x)
#         relu_coefs = theta.view(-1, self.channels, 2*self.k) * self.lambdas + self.init_v
#
#         # CxL -> CxLx1
#         x_perm = x.unsqueeze(-1)
#         output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
#         # CxLx2 -> CxL
#         result = torch.max(output, dim=-1)[0] #.permute(0, 1)
#
#         return result









# class DyReLUA(DyReLU):
#     def __init__(self, channels, reduction=4, k=2, conv_type='1d'):
#         super(DyReLUA, self).__init__(channels, reduction, k, conv_type)
#         self.fc2 = nn.Linear(channels // reduction, 2*k)
#
#     def forward(self, x):
#         assert x.shape[1] == self.channels
#         theta = self.get_relu_coefs(x)
#
#         relu_coefs = theta.view(-1, 2*self.k) * self.lambdas + self.init_v
#         # CxL -> CxLx1
#         x_perm = x.unsqueeze(-1)
#         output = x_perm * relu_coefs[:, :self.k] + relu_coefs[:, self.k:]
#         # CxLx2 -> CxL
#         result = torch.max(output, dim=-1)[0]
#
#         return result

