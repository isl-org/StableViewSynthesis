import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

import torch_scatter
import torch_geometric.nn as gnn
import torch_geometric.utils

from modules_gnn import *

sys.path.append("../")
import ext


class PointNetBase(nn.Module):
    def __init__(self, cat_dirs_mode="none"):
        super().__init__()
        self.cat_dirs_mode = cat_dirs_mode

    def _spherical_to_cartesian(self, spherical):
        vec = torch.empty(
            (spherical.shape[0], 3),
            dtype=spherical.dtype,
            device=spherical.device,
        )
        vec[:, 0] = torch.sin(spherical[:, 1]) * torch.cos(spherical[:, 0])
        vec[:, 1] = torch.sin(spherical[:, 1]) * torch.sin(spherical[:, 0])
        vec[:, 2] = torch.cos(spherical[:, 1])
        return vec

    def _cartesian_to_spherical(self, cartesian):
        sph = torch.empty(
            (cartesian.shape[0], 2),
            dtype=cartesian.dtype,
            device=cartesian.device,
        )
        sph[:, 0] = torch.atan2(cartesian[:, 1], cartesian[:, 0])
        sph[:, 1] = torch.acos(cartesian[:, 2])
        return sph

    def cat_dirs(self, x, dirs):
        if self.cat_dirs_mode == "spherical":
            with torch.no_grad():
                sph0 = self._cartesian_to_spherical(dirs[:, :2])
                sph1 = self._cartesian_to_spherical(dirs[:, 2:])
            # # test with cart_to_sph
            # tmp0 = self._spherical_to_cartesian(sph0)
            # tmp1 = self._spherical_to_cartesian(sph1)
            # if not (
            #     torch.allclose(dirs[:, :2], tmp0)
            #     and torch.allclose(dirs[:, 2:], tmp1)
            # ):
            #     raise Exception("nope")
            x = torch.cat((x, sph0, sph1), dim=1)
            del sph0, sph1
        elif self.cat_dirs_mode == "cartesian":
            x = torch.cat((x, dirs[:, :2], dirs[:, 2:]), dim=1)
        elif self.cat_dirs_mode == "cartesian_diff":
            with torch.no_grad():
                dir_diff = dirs[:, :2] - dirs[:, 2:]
            x = torch.cat((x, dir_diff), dim=1)
            del dir_diff
        elif self.cat_dirs_mode != "none":
            raise Exception("invalid cat_in_features")
        return x

    def forward(x, **kwargs):
        return x


class PointNetAvg(PointNetBase):
    def __init__(self, net=None, avg_mode="mean", net_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.net = net
        self.avg_mode = avg_mode

        if avg_mode == "softmax":
            self.feat_lin = nn.Linear(net_channels, net_channels)
            self.weight_lin = nn.Linear(net_channels, 1)

    def forward(
        self, x, point_key=None, point_edges=None, point_dirs=None, **kwargs
    ):
        x = self.cat_dirs(x, point_dirs)

        if self.net is not None:
            x = self.net(x, point_edges)

        if self.avg_mode == "mean":
            x = torch_scatter.segment_csr(x, point_key, reduce="mean")
        elif self.avg_mode == "dirs":
            with torch.no_grad():
                weight = (point_dirs[:, :3] * point_dirs[:, 3:]).sum(dim=1)
                weight = torch.clamp(weight, 0.01, 1)
                weight_sum = torch_scatter.segment_csr(
                    weight, point_key, reduce="sum"
                )
                weight /= torch_scatter.gather_csr(weight_sum, point_key)
            x = weight.view(-1, 1) * x
            x = torch_scatter.segment_csr(x, point_key, reduce="sum")
        elif self.avg_mode == "softmax":
            weight = self.weight_lin(x)
            weight = softmax_csr(weight, point_key)
            x = self.feat_lin(x)
            x = weight * x
            x = torch_scatter.segment_csr(x, point_key, reduce="sum")
        else:
            raise Exception("invalid avg_mode")
        return x


class PointWeighted(PointNetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, point_key=None, **kwargs):
        weights = x[:, -1:]
        weights = softmax_csr(weights, point_key)
        x = weights * x[:, :-1]
        x = torch_scatter.segment_csr(x, point_key, reduce="sum")
        return x


class PointNetCat(PointNetBase):
    def __init__(self, n_cat):
        super().__init__()
        self.n_cat = n_cat

    def forward(
        self, x, point_key=None, point_edges=None, point_dirs=None, **kwargs
    ):
        return ext.mytorch.point_cat(x, point_key, self.n_cat)

