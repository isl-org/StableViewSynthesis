import torch

try:
    from . import ext_cpu
except ImportError:
    try:
        import ext_cpu
    except ImportError as err:
        print("[WARNING] could not load ext_cpu in generated_ext.py")
        print(err)
try:
    from . import ext_cuda
except ImportError:
    try:
        import ext_cuda
    except ImportError as err:
        print("[WARNING] could not load ext_cuda in generated_ext.py")
        print(err)

class MapToListNn(torch.autograd.Function):
  @staticmethod
  def forward(ctx, features, nbidx, srcpos, bs, nv, height, width):
    args = (features.contiguous(), nbidx.contiguous(), srcpos.contiguous(), bs, nv, height, width)
    if features.is_cuda:
      out = ext_cuda.map_to_list_nn_forward(*args)
    else:
      out = ext_cpu.map_to_list_nn_forward(*args)
    ctx.bs = bs
    ctx.nv = nv
    ctx.height = height
    ctx.width = width
    ctx.save_for_backward(nbidx, srcpos)
    return out

  @staticmethod
  def backward(ctx, grad_out):
    grad_features = None
    grad_nbidx = None
    grad_srcpos = None
    grad_bs = None
    grad_nv = None
    grad_height = None
    grad_width = None
    nbidx, srcpos, = ctx.saved_tensors
    bs = ctx.bs
    nv = ctx.nv
    height = ctx.height
    width = ctx.width
    args = (nbidx.contiguous(), srcpos.contiguous(), bs, nv, height, width, grad_out.contiguous())
    if nbidx.is_cuda:
      grad_features = ext_cuda.map_to_list_nn_backward(*args)
    else:
      grad_features = ext_cpu.map_to_list_nn_backward(*args)
    return grad_features, grad_nbidx, grad_srcpos, grad_bs, grad_nv, grad_height, grad_width

def map_to_list_nn(features, nbidx, srcpos, bs, nv, height, width):
  return MapToListNn.apply(features, nbidx, srcpos, bs, nv, height, width)


class MapToListBl(torch.autograd.Function):
  @staticmethod
  def forward(ctx, features, nbidx, srcpos, bs, nv, height, width):
    args = (features.contiguous(), nbidx.contiguous(), srcpos.contiguous(), bs, nv, height, width)
    if features.is_cuda:
      out = ext_cuda.map_to_list_bl_forward(*args)
    else:
      out = ext_cpu.map_to_list_bl_forward(*args)
    ctx.bs = bs
    ctx.nv = nv
    ctx.height = height
    ctx.width = width
    ctx.save_for_backward(nbidx, srcpos)
    return out

  @staticmethod
  def backward(ctx, grad_out):
    grad_features = None
    grad_nbidx = None
    grad_srcpos = None
    grad_bs = None
    grad_nv = None
    grad_height = None
    grad_width = None
    nbidx, srcpos, = ctx.saved_tensors
    bs = ctx.bs
    nv = ctx.nv
    height = ctx.height
    width = ctx.width
    args = (nbidx.contiguous(), srcpos.contiguous(), bs, nv, height, width, grad_out.contiguous())
    if nbidx.is_cuda:
      grad_features = ext_cuda.map_to_list_bl_backward(*args)
    else:
      grad_features = ext_cpu.map_to_list_bl_backward(*args)
    return grad_features, grad_nbidx, grad_srcpos, grad_bs, grad_nv, grad_height, grad_width

def map_to_list_bl(features, nbidx, srcpos, bs, nv, height, width):
  return MapToListBl.apply(features, nbidx, srcpos, bs, nv, height, width)


class MapToListBlSeq(torch.autograd.Function):
  @staticmethod
  def forward(ctx, features, tgtidx, srcpos, out):
    args = (features.contiguous(), tgtidx.contiguous(), srcpos.contiguous(), out.contiguous())
    if features.is_cuda:
       ext_cuda.map_to_list_bl_seq_forward(*args)
    else:
       ext_cpu.map_to_list_bl_seq_forward(*args)
    return out

  @staticmethod
  def backward(ctx, ):
    grad_features = None
    grad_tgtidx = None
    grad_srcpos = None
    grad_out = None

    return grad_features, grad_tgtidx, grad_srcpos, grad_out

def map_to_list_bl_seq(features, tgtidx, srcpos, out):
  return MapToListBlSeq.apply(features, tgtidx, srcpos, out)


class ListToMap(torch.autograd.Function):
  @staticmethod
  def forward(ctx, features, tgtidx, bs, height, width):
    args = (features.contiguous(), tgtidx.contiguous(), bs, height, width)
    if features.is_cuda:
      out_sum, out_mask = ext_cuda.list_to_map_forward(*args)
    else:
      out_sum, out_mask = ext_cpu.list_to_map_forward(*args)
    ctx.bs = bs
    ctx.height = height
    ctx.width = width
    ctx.save_for_backward(tgtidx)
    return out_sum, out_mask

  @staticmethod
  def backward(ctx, grad_out_sum, grad_out_mask):
    grad_features = None
    grad_tgtidx = None
    grad_bs = None
    grad_height = None
    grad_width = None
    tgtidx, = ctx.saved_tensors
    bs = ctx.bs
    height = ctx.height
    width = ctx.width
    args = (grad_out_sum.contiguous(), tgtidx.contiguous(), bs, height, width)
    if grad_out_sum.is_cuda:
      grad_features = ext_cuda.list_to_map_backward(*args)
    else:
      grad_features = ext_cpu.list_to_map_backward(*args)
    return grad_features, grad_tgtidx, grad_bs, grad_height, grad_width

def list_to_map(features, tgtidx, bs, height, width):
  return ListToMap.apply(features, tgtidx, bs, height, width)


