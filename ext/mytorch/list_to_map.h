#pragma once

#include <co_math.h>
#include <common.h>
#include <torch_common.h>
#include <torch_tensor.h>

template <typename T>
struct ListToMapForward {
    const Tensor2<T> features;  // n_elems x channels
    const Tensor1<int> tgtidx;  // n_elems

    Tensor4<T> out_sum;   // batch_size x channels x height x width
    Tensor4<T> out_mask;  // batch_size x 1 x height x width

    ListToMapForward(const Tensor2<T> features, const Tensor1<int> tgtidx,
                     Tensor4<T> out_sum, Tensor4<T> out_mask)
        : features(features),
          tgtidx(tgtidx),
          out_sum(out_sum),
          out_mask(out_mask) {}

    CPU_GPU_FUNCTION void operator()(long idx) {
        // idx \in [0, n_elems]
        const long channels = out_sum.shape[1];
        const long height = out_sum.shape[2];
        const long width = out_sum.shape[3];

        // tgtidx = bidx * height * width + h * width + w
        const int tgtidx_ = tgtidx(idx);
        const int w = tgtidx_ % width;
        const int h = (tgtidx_ / width) % height;
        const int bidx = tgtidx_ / (height * width);

        for (long c = 0; c < channels; ++c) {
            co_atomic_add(out_sum.ptridx(bidx, c, h, w), features(idx, c));
        }
        out_mask(bidx, 0, h, w) = 1;
    }
};

template <typename T>
struct ListToMapBackward {
    const Tensor4<T> grad_out_sum;  // batch_size x channels x height x width
    const Tensor1<int> tgtidx;      // n_elems

    Tensor2<T> grad_features;  // n_elems x channels

    ListToMapBackward(const Tensor4<T> grad_out_sum, const Tensor1<int> tgtidx,
                      Tensor2<T> grad_features)
        : grad_out_sum(grad_out_sum),
          tgtidx(tgtidx),
          grad_features(grad_features) {}

    CPU_GPU_FUNCTION void operator()(long idx) {
        // idx \in [0, n_elems]
        const long channels = grad_out_sum.shape[1];
        const long height = grad_out_sum.shape[2];
        const long width = grad_out_sum.shape[3];

        // tgtidx = bidx * height * width + h * width + w
        const int tgtidx_ = tgtidx(idx);
        const int w = tgtidx_ % width;
        const int h = (tgtidx_ / width) % height;
        const int bidx = tgtidx_ / (height * width);

        for (long c = 0; c < channels; ++c) {
            grad_features(idx, c) = grad_out_sum(bidx, c, h, w);
        }
    }
};
