import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
import sys

import torch_scatter
import torch_geometric.nn as gnn
import torch_geometric.utils

sys.path.append("../")
import ext
import co

from modules_loss import *
from modules_unet import *


def _verify_and_get_test_set(worker):
    train_set = worker.get_train_set()
    if not (
        isinstance(train_set, co.mytorch.MultiDataset)
        and len(train_set.datasets) == 1
    ):
        raise Exception("invalid train_set, use only one")
    train_set = train_set.datasets[0]
    # enforce that train set and eval set are consistent
    # number of images, shape of images
    eval_set = worker.get_eval_sets()
    if len(eval_set) > 1:
        raise Exception("invalid number of eval_sets")
    eval_set = eval_set[0]
    if len(eval_set.src_im_paths) != len(train_set.src_im_paths):
        raise Exception("train and eval set src_im_paths do not match (length)")
    for p0, p1 in zip(eval_set.src_im_paths, train_set.src_im_paths):
        if p0 != p1:
            print(p0)
            print(p1)
            raise Exception(
                "train and eval set src_im_paths do not match (path)"
            )
    return train_set


def softmax_csr(x, key):
    x = x - torch_scatter.gather_csr(
        torch_scatter.segment_csr(x, key, reduce="max"), key
    )
    x = x.exp()
    x_sum = torch_scatter.gather_csr(
        torch_scatter.segment_csr(x, key, reduce="sum"), key
    )
    return x / (x_sum + 1e-16)


class GAT(nn.Module):
    def __init__(
        self, in_channels, channels=[8, 8, 16], heads=[4, 4, 1], mask=False
    ):
        super().__init__()
        self.mask = mask
        self.mods = nn.ModuleList()
        for channel, head in zip(channels, heads):
            self.mods.append(
                gnn.GATConv(
                    in_channels,
                    channel,
                    head,
                    concat=True,
                    add_self_loops=False,
                    bias=False,
                )
            )
            in_channels = head * channel

    def forward(
        self,
        x,
        ptx,
        bs,
        height,
        width,
        point_edges=None,
        pixel_tgt_idx=None,
        **kwargs,
    ):
        assert bs == 1
        # x map to list
        x = x.view(x.shape[1], height * width).transpose(1, 0)
        if self.mask:
            x_old = x
        # cat ptx, x to enable star graph
        x = torch.cat((ptx, x), dim=0)

        # run graph network
        for mod in self.mods:
            x = mod(x, point_edges)
            x = F.elu(x)

        # x list to map
        ptx = x[: -height * width]
        x = x[-height * width :]


        if self.mask:
            with torch.no_grad():
                mask = torch.zeros(
                    (x.shape[0], 1), dtype=torch.bool, device=x.device
                )
                mask[pixel_tgt_idx.long()] = 1
            x = torch.where(mask, x, x_old)

        x = x.view(bs, height, width, x.shape[1])
        x = x.permute(0, 3, 1, 2).contiguous()

        return x, ptx


class BaseEdge(nn.Module):
    def __init__(self, ch_feat_in=16, ch_edge_in=1, zero_in=True, mask=False):
        super().__init__()
        self.ch_feat_in = ch_feat_in
        self.ch_edge_in = ch_edge_in
        self.zero_in = zero_in
        self.mask = mask

    def net_forward(self, x, edge_index, edge_attr):
        pass

    def forward(
        self,
        x,
        ptx,
        bs,
        height,
        width,
        point_edges=None,
        point_src_dirs=None,
        point_tgt_dirs=None,
        pixel_tgt_idx=None,
        **kwargs,
    ):
        assert bs == 1
        x = x.view(x.shape[1], height * width).transpose(1, 0)
        if self.mask:
            x_old = x
        if self.zero_in:
            # cat ptx, x to enable star graph
            x = torch.cat((ptx, x), dim=0)
        else:
            x = torch.zeros(
                (ptx.shape[0] + x.shape[0], ptx.shape[1]), device=x.device
            )

        # run graph network
        point_dirs = torch.cat((point_src_dirs, point_tgt_dirs), dim=0)
        view_dot = (
            point_dirs[point_edges[0]] * point_dirs[point_edges[1]]
        ).sum(dim=1, keepdim=True)
        x = self.net_forward(x, point_edges, view_dot)

        # x list to map
        ptx = x[: -height * width]
        x = x[-height * width :]

        if self.mask:
            with torch.no_grad():
                mask = torch.zeros(
                    (x.shape[0], 1), dtype=torch.bool, device=x.device
                )
                mask[pixel_tgt_idx.long()] = 1
            x = torch.where(mask, x, x_old)

        x = x.view(bs, height, width, x.shape[1])
        x = x.permute(0, 3, 1, 2).contiguous()
        return x, ptx


class NNConvEdge(BaseEdge):
    def __init__(self, aggr="mean", **kwargs):
        super().__init__(**kwargs)

        nn1 = nn.Sequential(
            nn.Linear(self.ch_edge_in, 32),
            nn.ReLU(),
            nn.Linear(32, self.ch_feat_in * 32),
        )
        self.conv1 = gnn.NNConv(self.ch_feat_in, 32, nn1, aggr=aggr)
        nn2 = nn.Sequential(
            nn.Linear(self.ch_edge_in, 32), nn.ReLU(), nn.Linear(32, 32 * 16)
        )
        self.conv2 = gnn.NNConv(32, 16, nn2, aggr=aggr)

    def net_forward(self, x, edge_index, edge_attr):
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        return x


class BaseSoftMax(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.sm_out = nn.Linear(in_channels, 1)
        if out_channels > 0:
            self.feat_out = nn.Linear(in_channels, out_channels)
        else:
            self.feat_out = None

    def _net_forward(self, feat, point_key, point_edges):
        raise Exception("not implemented")

    def forward(
        self,
        x,
        ptx,
        bs,
        height,
        width,
        point_key=None,
        point_edges=None,
        pixel_tgt_idx=None,
        **kwargs,
    ):
        # cat ptx, x to feat
        assert bs == 1
        xch = x.shape[1]
        feat = x.permute(0, 2, 3, 1).view(height * width, xch)
        feat = feat[pixel_tgt_idx.long()]
        feat = torch_scatter.gather_csr(feat, point_key)
        feat = torch.cat((ptx, feat), dim=1)

        # eval network
        feat = self._net_forward(feat, point_key, point_edges)

        weight = softmax_csr(self.sm_out(feat), point_key)
        if self.feat_out:
            ptx = self.feat_out(feat)
        feat = weight * ptx
        feat = torch_scatter.segment_csr(feat, point_key, reduce="sum")

        # map feat to x
        feat, mask = ext.mytorch.list_to_map(
            feat, pixel_tgt_idx, bs, height, width
        )
        x = torch.where(mask > 0, feat, x)

        return x, ptx


class MLPSoftMax(BaseSoftMax):
    def __init__(self, in_channels, hidden_channels, n_mods, out_channels):
        super().__init__(hidden_channels, out_channels)
        in_channels *= 2
        mlp = []
        for _ in range(n_mods):
            mlp.append(nn.Linear(in_channels, hidden_channels))
            mlp.append(nn.ReLU())
            in_channels = hidden_channels
        self.mlp = nn.Sequential(*mlp)

    def _net_forward(self, feat, point_key, point_edges):
        return self.mlp(feat)


class GATSoftMax(BaseSoftMax):
    def __init__(
        self, in_channels, channels=[8, 8, 16], heads=[4, 4, 1], out_channels=0
    ):
        super().__init__(channels[-1] * heads[-1], out_channels)
        in_channels *= 2
        self.mods = nn.ModuleList()
        for channel, head in zip(channels, heads):
            self.mods.append(
                gnn.GATConv(in_channels, channel, head, concat=True)
            )
            in_channels = head * channel

    def _net_forward(self, x, point_key, point_edges):
        for mod in self.mods:
            x = mod(x, point_edges)
            x = F.elu(x)
        return x


class BaseAvg(nn.Module):
    def __init__(self, aggr="mean"):
        super().__init__()
        self.aggr = aggr

    def _net_forward(self, feat, point_key, point_edges):
        raise Exception("not implemented")

    def forward(
        self,
        x,
        ptx,
        bs,
        height,
        width,
        point_key=None,
        point_edges=None,
        pixel_tgt_idx=None,
        **kwargs,
    ):
        # cat ptx, x to feat
        assert bs == 1
        xch = x.shape[1]
        x_old = x
        x = x.permute(0, 2, 3, 1).view(height * width, xch)
        x = x[pixel_tgt_idx.long()]
        x = torch_scatter.gather_csr(x, point_key)
        x = torch.cat((ptx, x), dim=1)

        # eval network
        x = self._net_forward(x, point_key, point_edges)
        ptx = x
        x = torch_scatter.segment_csr(x, point_key, reduce=self.aggr)

        # map list to map
        x, mask = ext.mytorch.list_to_map(x, pixel_tgt_idx, bs, height, width)
        x = torch.where(mask > 0, x, x_old)

        return x, ptx


class MLPAvg(BaseAvg):
    def __init__(
        self, in_channels, hidden_channels, n_mods, out_channels, **kwargs
    ):
        super().__init__(**kwargs)
        in_channels *= 2
        mlp = []
        for _ in range(n_mods - 1):
            mlp.append(nn.Linear(in_channels, hidden_channels))
            mlp.append(nn.ReLU())
            in_channels = hidden_channels
        mlp.append(nn.Linear(in_channels, out_channels))
        self.mlp = nn.Sequential(*mlp)

    def _net_forward(self, feat, point_key, point_edges):
        return self.mlp(feat)


class GATAvg(BaseAvg):
    def __init__(
        self, in_channels, channels=[8, 8, 16], heads=[4, 4, 1], **kwargs
    ):
        super().__init__(**kwargs)
        in_channels *= 2
        self.mods = nn.ModuleList()
        for channel, head in zip(channels, heads):
            self.mods.append(
                gnn.GATConv(in_channels, channel, head, concat=True)
            )
            in_channels = head * channel

    def _net_forward(self, x, point_key, point_edges):
        for mod in self.mods:
            x = mod(x, point_edges)
            x = F.elu(x)
        return x


class BaseDir(nn.Module):
    def __init__(self, aggr="mean"):
        super().__init__()
        self.aggr = aggr

    def _net_forward(self, feat, point_key, point_edges):
        raise Exception("not implemented")

    def forward(
        self,
        x,
        ptx,
        bs,
        height,
        width,
        point_key=None,
        point_edges=None,
        point_src_dirs=None,
        point_tgt_dirs=None,
        pixel_tgt_idx=None,
        **kwargs,
    ):
        # eval network
        point_tgt_dirs = point_tgt_dirs[pixel_tgt_idx.long()]
        point_tgt_dirs = torch_scatter.gather_csr(point_tgt_dirs, point_key)
        ptx = torch.cat((ptx, point_src_dirs, point_tgt_dirs), dim=1)
        ptx = self._net_forward(ptx, point_key, point_edges)
        ptx = torch_scatter.segment_csr(ptx, point_key, reduce=self.aggr)

        # map list to map
        ptx, mask = ext.mytorch.list_to_map(
            ptx, pixel_tgt_idx, bs, height, width
        )
        x = torch.where(mask > 0, ptx, x)

        return x, ptx


class MLPDir(BaseDir):
    def __init__(
        self, in_channels, hidden_channels, n_mods, out_channels, **kwargs
    ):
        super().__init__(**kwargs)
        in_channels += 6
        mlp = []
        for _ in range(n_mods - 1):
            mlp.append(nn.Linear(in_channels, hidden_channels))
            mlp.append(nn.ReLU())
            in_channels = hidden_channels
        mlp.append(nn.Linear(in_channels, out_channels))
        self.mlp = nn.Sequential(*mlp)

    def _net_forward(self, feat, point_key, point_edges):
        return self.mlp(feat)


class GATDir(BaseDir):
    def __init__(
        self, in_channels, channels=[8, 8, 16], heads=[4, 4, 1], **kwargs
    ):
        super().__init__(**kwargs)
        in_channels += 6
        self.mods = nn.ModuleList()
        for channel, head in zip(channels, heads):
            self.mods.append(
                gnn.GATConv(in_channels, channel, head, concat=True)
            )
            in_channels = head * channel

    def _net_forward(self, x, point_key, point_edges):
        for mod in self.mods:
            x = mod(x, point_edges)
            x = F.elu(x)
        return x


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        if len(args) == 1:
            return args[0]
        else:
            return args


class RefNet(nn.Module):
    def __init__(
        self,
        point_avg_mode="mean",
        nets=None,
        nets_residual=False,
        gnns=None,
        out_conv=None,
    ):
        super().__init__()
        self.point_avg_mode = point_avg_mode
        self.nets_residual = nets_residual
        self.nets = nets
        if gnns is not None and len(gnns) > len(nets):
            raise Exception("invalid number of gnns")
        self.gnns = gnns
        self.out_conv = out_conv

    def forward(self, ptx, bs, height, width, **kwargs):  # point features
        point_key = kwargs["point_key"]
        pixel_tgt_idx = kwargs["pixel_tgt_idx"]

        # average per point
        if self.point_avg_mode == "avg":
            x = torch_scatter.segment_csr(ptx, point_key, reduce="mean")
        elif self.point_avg_mode == "diravg":
            with torch.no_grad():
                point_tgt_dirs = kwargs["point_tgt_dirs"][pixel_tgt_idx.long()]
                point_tgt_dirs = torch_scatter.gather_csr(
                    point_tgt_dirs, point_key
                )
                weight = (kwargs["point_src_dirs"] * point_tgt_dirs).sum(dim=1)
                weight = torch.clamp(weight, 0.01, 1)
                weight_sum = torch_scatter.segment_csr(
                    weight, point_key, reduce="sum"
                )
                weight /= torch_scatter.gather_csr(weight_sum, point_key)

            x = weight.view(-1, 1) * ptx
            x = torch_scatter.segment_csr(x, point_key, reduce="sum")
        else:
            raise Exception("invalid avg_mode")

        # project to target
        x, mask = ext.mytorch.list_to_map(x, pixel_tgt_idx, bs, height, width)

        # run refinement network
        for sidx in range(len(self.nets)):
            # process per 3D point
            if self.gnns is not None and sidx < len(self.gnns):
                gnn = self.gnns[sidx]
                x, ptx = gnn(x, ptx, bs, height, width, **kwargs)

            unet = self.nets[sidx]
            if self.nets_residual:
                x = x + unet(x)
            else:
                x = unet(x)

        if self.out_conv:
            x = self.out_conv(x)

        return {"out": x, "mask": mask}


def get_gnn(name, params, in_channels):
    # gat ... channel + head + channel + head + mask...
    # mlpsm ... n_mods + hidden_channels + out_channels
    # gatsm ... n_mods + channel + head + ...  + out_channels
    # mlpavg ... n_mods + hidden_channels + out_channels
    # mlpmax ... n_mods + hidden_channels + out_channels
    # gatavg ... channel + head + ...
    # nnconv ... aggr + zero_in + mask
    if name == "gat":
        mask = int(params[-1]) != 0
        channels = list(map(int, params[:-1:2]))
        heads = list(map(int, params[1:-1:2]))
        logging.info(
            f"[NET][RefNet]   GAT(in_channels={in_channels}, channels={channels}, heads={heads}, mask={mask})"
        )
        gnn = GAT(
            in_channels=in_channels, channels=channels, heads=heads, mask=mask
        )
        in_channels = channels[-1] * heads[-1]
    elif name == "mlpsm":
        n_mods, hidden_channels, out_channels = list(map(int, params))
        logging.info(
            f"[NET][RefNet]   MLPSoftMax(in_channels={in_channels}, hidden_channels={hidden_channels}, n_mods={n_mods}, out_channels={out_channels})"
        )
        gnn = MLPSoftMax(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            n_mods=n_mods,
            out_channels=out_channels,
        )
        in_channels = out_channels if out_channels > 0 else in_channels
    elif name == "gatsm":
        out_channels = int(params[-1])
        channels = list(map(int, params[:-1:2]))
        heads = list(map(int, params[1:-1:2]))
        logging.info(
            f"[NET][RefNet]   GATSoftMax(in_channels={in_channels}, channels={channels}, heads={heads}, out_channels={out_channels})"
        )
        gnn = GATSoftMax(
            in_channels=in_channels,
            channels=channels,
            heads=heads,
            out_channels=out_channels,
        )
        in_channels = out_channels if out_channels > 0 else in_channels
    elif name == "mlpavg":
        n_mods, hidden_channels, out_channels = list(map(int, params))
        logging.info(
            f"[NET][RefNet]   MLPAvg(in_channels={in_channels}, hidden_channels={hidden_channels}, n_mods={n_mods}, out_channels={out_channels})"
        )
        gnn = MLPAvg(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            n_mods=n_mods,
            out_channels=out_channels,
        )
        in_channels = out_channels
    elif name == "mlpmax":
        n_mods, hidden_channels, out_channels = list(map(int, params))
        logging.info(
            f"[NET][RefNet]   MLPMax(in_channels={in_channels}, hidden_channels={hidden_channels}, n_mods={n_mods}, out_channels={out_channels})"
        )
        gnn = MLPAvg(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            n_mods=n_mods,
            out_channels=out_channels,
            aggr="max",
        )
        in_channels = out_channels
    elif name == "mlpdir":
        aggr = params[0]
        n_mods, hidden_channels, out_channels = list(map(int, params[1:]))
        logging.info(
            f"[NET][RefNet]   MLPDir(in_channels={in_channels}, hidden_channels={hidden_channels}, n_mods={n_mods}, out_channels={out_channels}, aggr={aggr})"
        )
        gnn = MLPDir(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            n_mods=n_mods,
            out_channels=out_channels,
            aggr=aggr,
        )
        in_channels = out_channels
    elif name == "gatavg":
        channels = list(map(int, params[::2]))
        heads = list(map(int, params[1::2]))
        logging.info(
            f"[NET][RefNet]   GATAvg(in_channels={in_channels}, channels={channels}, heads={heads})"
        )
        gnn = GATAvg(in_channels=in_channels, channels=channels, heads=heads)
        in_channels = channels[-1] * heads[-1]
    elif name == "gatdir":
        aggr = params[0]
        params = params[1:]
        channels = list(map(int, params[::2]))
        heads = list(map(int, params[1::2]))
        logging.info(
            f"[NET][RefNet]   GATDir(in_channels={in_channels}, channels={channels}, heads={heads}, aggr={aggr})"
        )
        gnn = GATDir(
            in_channels=in_channels, channels=channels, heads=heads, aggr=aggr
        )
        in_channels = channels[-1] * heads[-1]
    elif name == "nnconv":
        aggr = params[0]
        zero_in = int(params[1]) != 0
        mask = int(params[2]) != 0
        logging.info(
            f"[NET][RefNet]   NNConv(in_channels={in_channels}, aggr={aggr}, zero_in={zero_in}, mask={mask})"
        )
        gnn = NNConvEdge(
            ch_feat_in=in_channels, ch_edge_in=1, zero_in=zero_in, mask=mask
        )
        in_channels = 16
    else:
        raise Exception(f"invalid gnn {name}")
    return gnn, in_channels


def get_refnet_net(net_name, params, in_channels):
    if net_name == "id":
        logging.info(f"[NET][RefNet]   Identity()")
        return Identity(), in_channels
    elif net_name == "unet":
        depth, n_conv, channels = list(map(int, params))
        enc_channels = [channels * (2 ** idx) for idx in range(depth - 1)]
        enc_channels.append(enc_channels[-1])
        dec_channels = enc_channels[::-1][1:]
        logging.info(
            f"[NET][RefNet]   Unet(in_channels={in_channels}, enc_channels={enc_channels}, dec_channels={dec_channels}, n_conv={n_conv})"
        )
        unet = UNet(
            in_channels=in_channels,
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            out_channels=-1,
            n_enc_convs=n_conv,
            n_dec_convs=n_conv,
        )
        return unet, dec_channels[-1]
    else:
        raise Exception(f"invalid net_name {net_name}")


def get_ref_net(name, in_channels):
    # ref_net format
    # point_edges_mode . point+aux+data . point_avg_mode .
    #   type+n_seq+residual+net [ . type+gnn]
    #
    # net=id  ... id
    # net=unet ... unet + depth + n_conv + channels

    splits = name.split(".")
    point_edges_mode = splits[0]
    point_aux_data = splits[1]
    point_avg_mode = splits[2]
    logging.info(f"[NET][RefNet] point_edges_mode={point_edges_mode}")
    logging.info(f"[NET][RefNet] point_aux_data={point_aux_data}")
    logging.info(f"[NET][RefNet] point_avg_mode={point_avg_mode}")

    net_params = splits[3].split("+")
    ref_type = net_params[0]
    n_seq = int(net_params[1])
    nets_residual = int(net_params[2]) != 0
    net_name = net_params[3]
    net_params = net_params[4:]

    nets = nn.ModuleList()
    if ref_type == "shared":
        logging.info(
            f"[NET][RefNet] Shared {n_seq} nets, nets_residual={nets_residual}"
        )
        net, in_channels = get_refnet_net(net_name, net_params, in_channels)
        for _ in range(n_seq):
            nets.append(net)
    elif ref_type == "seq":
        logging.info(
            f"[NET][RefNet] Seq {n_seq} nets, nets_residual={nets_residual}"
        )
        for _ in range(n_seq):
            net, in_channels = get_refnet_net(net_name, net_params, in_channels)
            nets.append(net)
    else:
        raise Exception(f"invalid ref_type {ref_type}")

    if len(splits) == 5:
        gnn = splits[4].split("+")
        gnn_type = gnn[0]
        gnn_name = gnn[1]
        gnn_params = gnn[2:]
        gnns = nn.ModuleList()
        if gnn_type == "single":
            logging.info("[NET][RefNet] Single gnn")
            gnn, in_channels = get_gnn(gnn_name, gnn_params, in_channels)
            gnns.append(gnn)
        elif gnn_type == "shared":
            logging.info(f"[NET][RefNet] Shared {n_seq} gnns")
            gnn, in_channels = get_gnn(gnn_name, gnn_params, in_channels)
            for _ in range(n_seq):
                gnns.append(gnn)
        elif gnn_type == "seq":
            logging.info(f"[NET][RefNet] Seq {n_seq} gnns")
            for gnnidx in range(n_seq):
                gnn, in_channels = get_gnn(gnn_name, gnn_params, in_channels)
                gnns.append(gnn)
        else:
            raise Exception(f"invalid gnn type {gnn_type}")
    else:
        gnns = None

    if net_name.startswith("id"):
        logging.info(f"[NET][RefNet] no out_conv")
        out_conv = None
    else:
        logging.info(f"[NET][RefNet] out_conv({in_channels}, 3)")
        out_conv = nn.Conv2d(in_channels, 3, kernel_size=1, padding=0)

    ref_net = RefNet(
        point_avg_mode=point_avg_mode,
        nets=nets,
        nets_residual=nets_residual,
        gnns=gnns,
        out_conv=out_conv,
    )
    # print(ref_net)
    return ref_net


class Net(nn.Module):
    def __init__(self, enc_net, ref_net):
        super().__init__()
        self.enc_net = enc_net
        self.ref_net = ref_net

    def enc_forward(self, src_ims=None, src_ind=None):
        bs, nv, _, height, width = src_ims.shape
        features = self.enc_net(src_ims.view(bs * nv, *src_ims.shape[-3:]))
        return features.view(bs, nv, *features.shape[-3:])

    def ref_net_forward(self, ptx, bs, height, width, **kwargs):
        return self.ref_net(ptx, bs, height, width, **kwargs)

    def forward(self, **kwargs):
        x, src_ind = None, None
        if "src_ims" in kwargs:
            x = kwargs["src_ims"]
        if "src_ind" in kwargs:
            src_ind = kwargs["src_ind"]
        x = self.enc_forward(src_ims=x, src_ind=src_ind)

        # extract point data from srcs
        bs, nv, height, width = x.shape[0], x.shape[1], x.shape[-2], x.shape[-1]
        x = ext.mytorch.map_to_list_bl(
            x,
            kwargs["m2l_src_idx"],
            kwargs["m2l_src_pos"],
            bs,
            nv,
            height,
            width,
        )

        return self.ref_net_forward(x, bs, height, width, **kwargs)


def get_net(enc_net, ref_net, init_net_path=None):
    if enc_net in ["id", "identity"]:
        logging.info(f"[NET][EncNet] identity")
        enc_net = Identity()
        enc_channels = 3
    elif enc_net == "iddummy":
        logging.info(f"[NET][EncNet] iddummy")
        enc_net = nn.Conv2d(3, 3, kernel_size=1, padding=0, bias=False)
        enc_net.weight.data.fill_(0)
        enc_net.weight.data[0, 0, 0, 0] = 1
        enc_net.weight.data[1, 1, 0, 0] = 1
        enc_net.weight.data[2, 2, 0, 0] = 1
        enc_channels = 3
    elif enc_net == "resunet3":
        logging.info(f"[NET][EncNet] resunet3[.64]")
        enc_net = ResUNet(depth=3)
        enc_channels = 64
    elif enc_net == "resunet3.32":
        logging.info(f"[NET][EncNet] resunet3.32")
        enc_net = ResUNet(out_channels_0=32, depth=3)
        enc_channels = 32
    elif enc_net == "resunet3.16":
        logging.info(f"[NET][EncNet] resunet3.16")
        enc_net = ResUNet(out_channels_0=16, depth=3)
        enc_channels = 16
    elif enc_net == "vggunet16.3":
        logging.info(f"[NET][EncNet] vggunet16.3")
        enc_net = VGGUNet(net="vgg16", n_encoder_stages=3)
        enc_channels = 64
    else:
        raise Exception("invalid enc_net")

    ref_net = get_ref_net(ref_net, enc_channels)

    net = Net(enc_net=enc_net, ref_net=ref_net)

    # load net if exists
    if init_net_path and len(init_net_path) > 0:
        logging.info(f"[get_net] load init net from {init_net_path}")
        net_params = torch.load(init_net_path)
        net.load_state_dict(net_params)

    return net


class ParamNet(Net):
    def __init__(self, net, dset, eval_device):
        super().__init__(enc_net=None, ref_net=net.ref_net)
        self._create_params(net.enc_net, dset, eval_device)

    def _create_params(self, enc_net, dset, eval_device):
        tic = time.time()
        logging.info("[ParamNet] create params")
        src_im_paths = dset.src_im_paths
        params = None
        enc_net = enc_net.to(eval_device)
        enc_net.eval()
        with torch.no_grad():
            for idx, src_im_path in enumerate(src_im_paths):
                logging.info(
                    f"[ParamNet] create params {idx+1}/{len(src_im_paths)}"
                )
                im = dset.load_pad(src_im_path)[None]
                x = torch.from_numpy(im).to(eval_device)
                x = enc_net(x)
                if params is None:
                    params = torch.empty(
                        (len(src_im_paths), *x.shape[1:]),
                        dtype=x.dtype,
                        device="cpu",
                    )
                params[idx] = x[0].detach().to("cpu")
        self.params = nn.Parameter(params)
        logging.info(
            f"[ParamNet] done create params, took {time.time() - tic}[s]"
        )

    def to(self, *args, **kwargs):
        self.ref_net = self.ref_net.to(*args, **kwargs)
        return self

    def enc_forward(self, src_ims=None, src_ind=None):
        src_ind_device = src_ind.device
        src_ind = src_ind.to(self.params.device)

        sel_params = self.params[src_ind]

        sel_params = sel_params.to(src_ind_device)
        return sel_params


def get_param_net(worker, enc_net, ref_net, init_net_path=None):
    net = get_net(enc_net, ref_net, init_net_path=init_net_path)
    train_set = _verify_and_get_test_set(worker)
    return ParamNet(net, train_set, worker.eval_device)


class ImageNet(Net):
    def __init__(self, net, dset):
        super().__init__(enc_net=net.enc_net, ref_net=net.ref_net)
        self._create_params(net.enc_net, dset)

    def _create_params(self, enc_net, dset):
        tic = time.time()
        logging.info("[ImageNet] create params")
        src_im_paths = dset.src_im_paths
        params = None
        for idx, src_im_path in enumerate(src_im_paths):
            logging.info(
                f"[ImageNet] create params {idx+1}/{len(src_im_paths)}"
            )
            im = dset.load_pad(src_im_path)[None]
            x = torch.from_numpy(im)
            if params is None:
                params = torch.empty(
                    (len(src_im_paths), *x.shape[1:]),
                    dtype=x.dtype,
                    device="cpu",
                )
            params[idx] = x[0].detach().to("cpu")
        self.params = nn.Parameter(params)
        logging.info(
            f"[ImageNet] done create params, took {time.time() - tic}[s]"
        )

    def enc_forward(self, src_ims=None, src_ind=None):
        src_ind_device = src_ind.device
        src_ind = src_ind.to(self.params.device)
        sel_params = self.params[src_ind]
        sel_params = sel_params.to(src_ind_device)
        bs, nv, _, height, width = sel_params.shape
        features = self.enc_net(sel_params.view(bs * nv, *src_ims.shape[-3:]))
        return features.view(bs, nv, *features.shape[-3:])


def get_image_net(worker, enc_net, ref_net, init_net_path=None):
    net = get_net(enc_net, ref_net, init_net_path=init_net_path)
    train_set = _verify_and_get_test_set(worker)
    return ImageNet(net, train_set)


class GlobalSONet(Net):
    def __init__(self, net, dset):
        super().__init__(enc_net=net.enc_net, ref_net=net.ref_net)
        self._create_params(dset)

    def _create_params(self, dset):
        tic = time.time()
        logging.info("[GlobalSONet] create params")
        src_im_paths = dset.src_im_paths
        n = len(src_im_paths)
        self.scale_params = nn.Parameter(torch.ones((n,)))
        self.offset_params = nn.Parameter(torch.zeros((n,)))
        logging.info(
            f"[GlobalSONet] done create params, took {time.time() - tic}[s]"
        )

    def enc_forward(self, src_ims=None, src_ind=None):
        bs, nv, _, height, width = src_ims.shape
        scale_params = self.scale_params[src_ind].view(bs, nv, 1, 1, 1)
        offset_params = self.offset_params[src_ind].view(bs, nv, 1, 1, 1)
        src_ims = scale_params * src_ims + offset_params
        src_ims = torch.clamp(src_ims, -1, 1)
        features = self.enc_net(src_ims.view(bs * nv, *src_ims.shape[-3:]))
        return features.view(bs, nv, *features.shape[-3:])


def get_globalso_net(worker, enc_net, ref_net, init_net_path=None):
    net = get_net(enc_net, ref_net, init_net_path=init_net_path)
    train_set = _verify_and_get_test_set(worker)
    return GlobalSONet(net, train_set)


class WeightNet(Net):
    def __init__(self, net, dset, pixelwise=True):
        super().__init__(enc_net=net.enc_net, ref_net=net.ref_net)
        self.pixelwise = pixelwise
        self._create_params(dset, pixelwise)

    def _create_params(self, dset, pixelwise):
        tic = time.time()
        logging.info("[WeightNet] create params")
        src_im_paths = dset.src_im_paths
        n = len(src_im_paths)
        if pixelwise:
            logging.info("[WeightNet]   pixelwise")
            im = dset.load_pad(src_im_paths[0])
            self.params = nn.Parameter(
                torch.ones((n, 1, im.shape[-2], im.shape[-1]))
            )
        else:
            logging.info("[WeightNet]   global")
            self.params = nn.Parameter(torch.ones((n, 1, 1, 1)))
        logging.info(
            f"[WeightNet] done create params, took {time.time() - tic}[s]"
        )

    def enc_forward(self, src_ims=None, src_ind=None):
        bs, nv, _, height, width = src_ims.shape
        features = self.enc_net(src_ims.view(bs * nv, *src_ims.shape[-3:]))
        features = features.view(bs, nv, *features.shape[-3:])
        params = self.params[src_ind]
        if not self.pixelwise:
            params = params.expand(-1, -1, -1, *features.shape[-2:])
        features = torch.cat((features, params), dim=2)
        return features


def get_weight_net(
    worker, enc_net, ref_net, init_net_path=None, pixelwise=True
):
    net = get_net(enc_net, ref_net, init_net_path=init_net_path)
    train_set = _verify_and_get_test_set(worker)
    return WeightNet(net, train_set, pixelwise=pixelwise)
