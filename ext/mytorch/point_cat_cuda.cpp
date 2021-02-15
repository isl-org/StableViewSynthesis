#include "point_cat.h"

void point_cat_forward_kernel(at::Tensor in_features, at::Tensor key, int n_cat,
                              at::Tensor out_features);

at::Tensor point_cat_forward(at::Tensor in_features, at::Tensor key,
                             int n_cat) {
    CHECK_INPUT_CUDA_DIM(in_features, 2);
    const long channels = in_features.size(1);
    CHECK_INPUT_CUDA_DIM(key, 1);
    const long prefix_sum = key.size(0);

    auto out_features =
        torch::zeros({prefix_sum - 1, n_cat * channels},
                     torch::TensorOptions()
                         .dtype(in_features.dtype())
                         .device(in_features.device().type())
                         .requires_grad(in_features.requires_grad()));

    point_cat_forward_kernel(in_features, key, n_cat, out_features);

    return out_features;
}

void point_cat_backward_kernel(at::Tensor grad_out_features, at::Tensor key,
                               int n_cat, at::Tensor grad_in_features);

at::Tensor point_cat_backward(at::Tensor grad_out_features, at::Tensor key,
                              int n_cat, int nelems) {
    CHECK_INPUT_CUDA_DIM(grad_out_features, 2);
    const long prefix_sum = grad_out_features.size(0) + 1;
    const long channels = grad_out_features.size(1) / n_cat;
    CHECK_INPUT_CUDA_DIM(key, 1);
    AT_ASSERTM(key.size(0) == 1 * prefix_sum,
               "prefix_sum of key does not match");

    auto grad_in_features =
        torch::zeros({nelems, channels},
                     torch::TensorOptions()
                         .dtype(grad_out_features.dtype())
                         .device(grad_out_features.device().type())
                         .requires_grad(grad_out_features.requires_grad()));

    point_cat_backward_kernel(grad_out_features, key, n_cat, grad_in_features);

    return grad_in_features;
}

void point_cat_pybind_init(py::module& m) {
    m.def("point_cat_forward", &point_cat_forward);
    m.def("point_cat_backward", &point_cat_backward);
}
