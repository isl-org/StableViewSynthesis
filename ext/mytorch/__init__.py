import torch

from .generated_ext import *

try:
    from . import ext_cpu
except ImportError:
    try:
        import ext_cpu
    except ImportError as err:
        print("[WARNING] could not load ext_cpu in __init__.py")
        print(err)
try:
    from . import ext_cuda
except ImportError:
    try:
        import ext_cuda
    except ImportError as err:
        print("[WARNING] could not load ext_cuda in __init__.py")
        print(err)


class PointCat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, in_features, key, n_cat):
        args = (in_features.contiguous(), key.contiguous(), n_cat)
        if in_features.is_cuda:
            out_features = ext_cuda.point_cat_forward(*args)
        else:
            out_features = ext_cpu.point_cat_forward(*args)
        ctx.n_cat = n_cat
        ctx.nelems = in_features.shape[0]
        ctx.save_for_backward(key)
        return out_features

    @staticmethod
    def backward(ctx, grad_out_features):
        grad_in_features = None
        key, = ctx.saved_tensors
        n_cat = ctx.n_cat
        nelems = ctx.nelems
        args = (grad_out_features.contiguous(), key.contiguous(), n_cat, nelems)
        if grad_out_features.is_cuda:
            grad_in_features = ext_cuda.point_cat_backward(*args)
        else:
            grad_in_features = ext_cpu.point_cat_backward(*args)
        return grad_in_features, None, None


def point_cat(in_features, key, n_cat):
    return PointCat.apply(in_features, key, n_cat)
