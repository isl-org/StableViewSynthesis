import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging


class ImNormalizer(object):
    def __init__(self, in_fmt="-11"):
        self.in_fmt = in_fmt
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def apply(self, x):
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        if self.in_fmt == "-11":
            x = (x + 1) / 2
        elif self.in_fmt != "01":
            raise Exception("invalid input format")
        return (x - self.mean) / self.std


class VGGUNet(nn.Module):
    def __init__(
        self, net="vgg16", pool="average", n_encoder_stages=3, n_decoder_convs=2
    ):
        super().__init__()

        if net == "vgg16":
            vgg = torchvision.models.vgg16(pretrained=True).features
        elif net == "vgg19":
            vgg = torchvision.models.vgg19(pretrained=True).features
        else:
            raise Exception("invalid vgg net")

        self.normalizer = ImNormalizer()

        encs = []
        enc = []
        encs_channels = []
        channels = -1
        for mod in vgg:
            if isinstance(mod, nn.Conv2d):
                channels = mod.out_channels

            if isinstance(mod, nn.MaxPool2d):
                encs.append(nn.Sequential(*enc))
                encs_channels.append(channels)
                n_encoder_stages -= 1
                if n_encoder_stages <= 0:
                    break
                if pool == "average":
                    enc = [
                        nn.AvgPool2d(
                            kernel_size=2, stride=2, padding=0, ceil_mode=False
                        )
                    ]
                elif pool == "max":
                    enc = [
                        nn.MaxPool2d(
                            kernel_size=2, stride=2, padding=0, ceil_mode=False
                        )
                    ]
                else:
                    raise Exception("invalid pool")
            else:
                enc.append(mod)
        self.encs = nn.ModuleList(encs)

        cin = encs_channels[-1] + encs_channels[-2]
        decs = []
        for idx, cout in enumerate(reversed(encs_channels[:-1])):
            decs.append(self._dec(cin, cout, n_convs=n_decoder_convs))
            cin = cout + encs_channels[max(-idx - 3, -len(encs_channels))]
        self.decs = nn.ModuleList(decs)

    def _dec(self, channels_in, channels_out, n_convs=2):
        mods = []
        for _ in range(n_convs):
            mods.append(
                nn.Conv2d(
                    channels_in,
                    channels_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
            mods.append(nn.ReLU())
            channels_in = channels_out
        return nn.Sequential(*mods)

    def forward(self, x):
        x = self.normalizer.apply(x)

        feats = []
        for enc in self.encs:
            x = enc(x)
            feats.append(x)

        for dec in self.decs:
            x0 = feats.pop()
            x1 = feats.pop()
            x0 = F.interpolate(
                x0, size=(x1.shape[2], x1.shape[3]), mode="nearest"
            )
            x = torch.cat((x0, x1), dim=1)
            x = dec(x)
            feats.append(x)

        x = feats.pop()
        return x


class ResUNet(nn.Module):
    def __init__(
        self, out_channels_0=64, out_channels=-1, depth=5, resnet="resnet18"
    ):
        super().__init__()

        if resnet == "resnet18":
            resnet = torchvision.models.resnet18(pretrained=True)
        else:
            raise Exception("invalid resnet model")

        self.normalizer = ImNormalizer()

        if depth < 1 or depth > 5:
            raise Exception("invalid depth of UNet")

        encs = nn.ModuleList()
        enc_translates = nn.ModuleList()
        decs = nn.ModuleList()
        enc_channels = 0
        if depth == 5:
            encs.append(resnet.layer4)
            enc_translates.append(self.convrelu(512, 512, 1))
            enc_channels = 512
        if depth >= 4:
            encs.append(resnet.layer3)
            enc_translates.append(self.convrelu(256, 256, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 256, 256))
            enc_channels = 256
        if depth >= 3:
            encs.append(resnet.layer2)
            enc_translates.append(self.convrelu(128, 128, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 128, 128))
            enc_channels = 128
        if depth >= 2:
            encs.append(nn.Sequential(resnet.maxpool, resnet.layer1))
            enc_translates.append(self.convrelu(64, 64, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 64, 64))
            enc_channels = 64
        if depth >= 1:
            encs.append(nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu))
            enc_translates.append(self.convrelu(64, 64, 1))
            if enc_channels > 0:
                decs.append(self.convrelu(enc_channels + 64, 64))
            enc_channels = 64
        enc_translates.append(
            nn.Sequential(self.convrelu(3, 64), self.convrelu(64, 64))
        )
        decs.append(self.convrelu(enc_channels + 64, out_channels_0))

        self.encs = nn.ModuleList(reversed(encs))
        self.enc_translates = nn.ModuleList(reversed(enc_translates))
        self.decs = nn.ModuleList(decs)

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        if out_channels <= 0:
            self.out_conv = None
        else:
            self.out_conv = nn.Conv2d(
                out_channels_0, out_channels, kernel_size=1, padding=0
            )

    def convrelu(self, in_channels, out_channels, kernel_size=3, padding=None):
        if padding is None:
            padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    # disable batchnorm learning in self.encs
    def train(self, mode=True):
        super().train(mode=mode)
        if not mode:
            return
        for mod in self.encs.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad_(False)

    def forward(self, x):
        x = self.normalizer.apply(x)

        outs = [self.enc_translates[0](x)]
        for enc, enc_translates in zip(self.encs, self.enc_translates[1:]):
            x = enc(x)
            outs.append(enc_translates(x))

        for dec in self.decs:
            x0, x1 = outs.pop(), outs.pop()
            x = torch.cat((self.upsample(x0), x1), dim=1)
            x = dec(x)
            outs.append(x)
        x = outs.pop()

        if self.out_conv:
            x = self.out_conv(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        enc_channels=[64, 128, 256],
        dec_channels=[128, 64],
        out_channels=3,
        n_enc_convs=2,
        n_dec_convs=2,
    ):
        super().__init__()

        self.encs = nn.ModuleList()
        self.enc_translates = nn.ModuleList()
        pool = False
        for enc_channel in enc_channels:
            stage = self.create_stage(
                in_channels, enc_channel, n_enc_convs, pool
            )
            self.encs.append(stage)
            translate = nn.Conv2d(enc_channel, enc_channel, kernel_size=1)
            self.enc_translates.append(translate)
            in_channels, pool = enc_channel, True

        self.decs = nn.ModuleList()
        for idx, dec_channel in enumerate(dec_channels):
            in_channels = enc_channels[-idx - 1] + enc_channels[-idx - 2]
            stage = self.create_stage(
                in_channels, dec_channel, n_dec_convs, False
            )
            self.decs.append(stage)

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        if out_channels <= 0:
            self.out_conv = None
        else:
            self.out_conv = nn.Conv2d(
                dec_channels[-1], out_channels, kernel_size=1, padding=0
            )

    def convrelu(self, in_channels, out_channels, kernel_size=3, padding=None):
        if padding is None:
            padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    def create_stage(self, in_channels, out_channels, n_convs, pool):
        mods = []
        if pool:
            mods.append(nn.AvgPool2d(kernel_size=2))
        for _ in range(n_convs):
            mods.append(self.convrelu(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*mods)

    def forward(self, x):
        outs = []
        for enc, enc_translates in zip(self.encs, self.enc_translates):
            x = enc(x)
            outs.append(enc_translates(x))

        for dec in self.decs:
            x0, x1 = outs.pop(), outs.pop()
            x = torch.cat((self.upsample(x0), x1), dim=1)
            x = dec(x)
            outs.append(x)

        x = outs.pop()
        if self.out_conv:
            x = self.out_conv(x)
        return x


class SeqUNet(nn.Module):
    def __init__(
        self,
        n_seq,
        in_channels,
        enc_channels=[64, 128, 256],
        dec_channels=[128, 64],
        out_channels=3,
        n_enc_convs=2,
        n_dec_convs=2,
        refine_add=False,
    ):
        super().__init__()
        self.n_seq = n_seq
        self.refine_add = refine_add

        self.unets = nn.ModuleList()
        for _ in range(n_seq):
            self.unets.append(
                UNet(
                    in_channels=in_channels,
                    enc_channels=enc_channels,
                    dec_channels=dec_channels,
                    out_channels=-1,
                    n_enc_convs=n_enc_convs,
                    n_dec_convs=n_dec_convs,
                )
            )
            in_channels = dec_channels[-1]

        if out_channels <= 0:
            self.out_conv = None
        else:
            self.out_conv = nn.Conv2d(
                dec_channels[-1], out_channels, kernel_size=1, padding=0
            )

    def forward(self, x):
        for unet in self.unets:
            if self.refine_add:
                x = x + unet(x)
            else:
                x = unet(x)

        if self.out_conv:
            x = self.out_conv(x)
        return x


def get_unet(unet, in_channels):
    slen = 0
    if unet.startswith("seq"):
        refine_add = "seqadd" in unet
        slen = 6 if refine_add else 3
        n_seq = int(unet[slen : slen + 1])
        slen = unet.find("unet")
    plen = slen + 4
    unet = unet[plen:]
    depth, n_conv, channels = unet.split(".")
    depth, n_conv, channels = int(depth), int(n_conv), int(channels)
    enc_channels = [channels * (2 ** idx) for idx in range(depth - 1)]
    enc_channels.append(enc_channels[-1])
    dec_channels = enc_channels[::-1][1:]
    n_enc_convs = n_conv
    n_dec_convs = n_conv
    logging.info(f"[UNet]         enc_channels={enc_channels}")
    logging.info(f"[UNet]         dec_channels={dec_channels}")
    logging.info(f"[UNet]         n_enc_convs={n_enc_convs}")
    logging.info(f"[UNet]         n_dec_convs={n_dec_convs}")
    if slen > 0:
        logging.info(f"[UNet]         n_seq={n_seq}")
        logging.info(f"[UNet]         refine_add={refine_add}")
        return SeqUNet(
            n_seq=n_seq,
            in_channels=in_channels,
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            out_channels=3,
            n_enc_convs=n_enc_convs,
            n_dec_convs=n_dec_convs,
            refine_add=refine_add,
        )
    else:
        return UNet(
            in_channels=in_channels,
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            out_channels=3,
            n_enc_convs=n_enc_convs,
            n_dec_convs=n_dec_convs,
        )
