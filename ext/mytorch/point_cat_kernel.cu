#include <common_cuda.h>
#include <torch_common.h>
#include "point_cat.h"

void point_cat_forward_kernel(at::Tensor in_features, at::Tensor key, int n_cat,
                              at::Tensor out_features) {
    AT_DISPATCH_FLOATING_TYPES(
        out_features.scalar_type(), "PointCatForward", ([&] {
            iterate_cuda(
                PointCatForward<scalar_t>(torch2co2<scalar_t>(in_features),
                                          torch2co1<long>(key), n_cat,
                                          torch2co2<scalar_t>(out_features)),
                key.size(0) - 1);
        }));
}
void point_cat_backward_kernel(at::Tensor grad_out_features, at::Tensor key,
                               int n_cat, at::Tensor grad_in_features) {
    AT_DISPATCH_FLOATING_TYPES(
        grad_in_features.scalar_type(), "PointCatBackward", ([&] {
            iterate_cuda(PointCatBackward<scalar_t>(
                             torch2co2<scalar_t>(grad_out_features),
                             torch2co1<long>(key), n_cat,
                             torch2co2<scalar_t>(grad_in_features)),
                         grad_out_features.size(0));
        }));
}
