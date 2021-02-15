import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import sys

sys.path.append("../")
import config


class VGGPerceptualLoss(nn.Module):
    def __init__(self, inp_scale="-11"):
        super().__init__()
        self.inp_scale = inp_scale
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.vgg = torchvision.models.vgg19(pretrained=True).features

    def forward(self, es, ta):
        self.vgg = self.vgg.to(es.device)
        self.mean = self.mean.to(es.device)
        self.std = self.std.to(es.device)

        if self.inp_scale == "-11":
            es = (es + 1) / 2
            ta = (ta + 1) / 2
        elif self.inp_scale != "01":
            raise Exception("invalid input scale")
        es = (es - self.mean) / self.std
        ta = (ta - self.mean) / self.std

        loss = [torch.abs(es - ta).mean()]
        for midx, mod in enumerate(self.vgg):
            es = mod(es)
            with torch.no_grad():
                ta = mod(ta)

            if midx == 3:
                lam = 1
                loss.append(torch.abs(es - ta).mean() * lam)
            elif midx == 8:
                lam = 0.75
                loss.append(torch.abs(es - ta).mean() * lam)
            elif midx == 13:
                lam = 0.5
                loss.append(torch.abs(es - ta).mean() * lam)
            elif midx == 22:
                lam = 0.5
                loss.append(torch.abs(es - ta).mean() * lam)
            elif midx == 31:
                lam = 1
                loss.append(torch.abs(es - ta).mean() * lam)
                break
        return loss


class LPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = None
        self.clip = True

    def forward(self, es, ta):
        if self.mod is None:
            sys.path.append(str(config.lpips_root))
            import PerceptualSimilarity.models as ps

            self.mod = ps.PerceptualLoss()

        if self.clip:
            es = torch.clamp(es, -1, 1)
        out = self.mod(es, ta, normalize=False)

        return out.mean()


# https://github.com/nianticlabs/monodepth2/blob/master/layers.py#L218
class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (
            sigma_x + sigma_y + self.C2
        )

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class SSIML1Loss(nn.Module):
    def __init__(self, alpha=0.85):
        super().__init__()
        self.alpha = alpha
        self.ssim = SSIMLoss()

    def forward(self, es, ta):
        es = (es + 1) / 2
        ta = (ta + 1) / 2
        ssim = self.alpha * self.ssim(es, ta).mean()
        l1 = (1 - self.alpha) * torch.abs(es - ta).mean()
        loss = [ssim, l1]
        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, es, ta):
        return [F.l1_loss(es, ta)]
