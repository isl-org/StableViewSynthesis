#include <common_cpu.h>
#include "point_cat.h"

at::Tensor point_cat_forward(at::Tensor in_features, at::Tensor key,
                             int n_cat) {
    CHECK_INPUT_CPU_DIM(in_features, 2);
    const long channels = in_features.size(1);
    CHECK_INPUT_CPU_DIM(key, 1);
    const long prefix_sum = key.size(0);

    auto out_features =
        torch::zeros({prefix_sum - 1, n_cat * channels},
                     torch::TensorOptions()
                         .dtype(in_features.dtype())
                         .device(in_features.device().type())
                         .requires_grad(in_features.requires_grad()));

    AT_DISPATCH_FLOATING_TYPES(
        out_features.scalar_type(), "PointCatForward", ([&] {
            iterate_cpu(
                PointCatForward<scalar_t>(torch2co2<scalar_t>(in_features),
                                          torch2co1<long>(key), n_cat,
                                          torch2co2<scalar_t>(out_features)),
                key.size(0) - 1);
        }));

    return out_features;
}

at::Tensor point_cat_backward(at::Tensor grad_out_features, at::Tensor key,
                              int n_cat, int nelems) {
    CHECK_INPUT_CPU_DIM(grad_out_features, 2);
    const long prefix_sum = grad_out_features.size(0) + 1;
    const long channels = grad_out_features.size(1) / n_cat;
    CHECK_INPUT_CPU_DIM(key, 1);
    AT_ASSERTM(key.size(0) == 1 * prefix_sum,
               "prefix_sum of key does not match");

    auto grad_in_features =
        torch::zeros({nelems, channels},
                     torch::TensorOptions()
                         .dtype(grad_out_features.dtype())
                         .device(grad_out_features.device().type())
                         .requires_grad(grad_out_features.requires_grad()));

    AT_DISPATCH_FLOATING_TYPES(
        grad_in_features.scalar_type(), "PointCatBackward", ([&] {
            iterate_cpu(PointCatBackward<scalar_t>(
                            torch2co2<scalar_t>(grad_out_features),
                            torch2co1<long>(key), n_cat,
                            torch2co2<scalar_t>(grad_in_features)),
                        grad_out_features.size(0));
        }));

    return grad_in_features;
}

void point_cat_pybind_init(py::module& m) {
    m.def("point_cat_forward", &point_cat_forward);
    m.def("point_cat_backward", &point_cat_backward);
}
