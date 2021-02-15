#pragma once

#include <co_math.h>
#include <common.h>
#include <torch_common.h>
#include <torch_tensor.h>

template <typename T>
struct MapToListNnForward {
    const Tensor5<T>
        features;  // batch_size x n_views x channels x height x width
    const Tensor1<int> nbidx;  // n_elems
    const Tensor2<T> srcpos;   // n_elems x 2

    Tensor2<T> out;  // n_elems x channels

    MapToListNnForward(const Tensor5<T> features, const Tensor1<int> nbidx,
                       const Tensor2<T> srcpos, Tensor2<T> out)
        : features(features), nbidx(nbidx), srcpos(srcpos), out(out) {}

    CPU_GPU_FUNCTION void operator()(long idx) {
        // idx \in [0, n_elems]
        const long batch_size = features.shape[0];
        const long channels = features.shape[2];

        const int bidx = nbidx(idx) % batch_size;
        const int vidx = nbidx(idx) / batch_size;

        // Pixel center is assumed to be at x+0.5, y+0.5
        // Hence, floor (int cast) gets the nearest neighbour
        const int h = srcpos(idx, 0);
        const int w = srcpos(idx, 1);
        // We further assume that srcpos are all valid (inside map)

        for (long c = 0; c < channels; ++c) {
            out(idx, c) = features(bidx, vidx, c, h, w);
        }
    }
};

template <typename T>
struct MapToListNnBackward {
    const Tensor1<int> nbidx;  // n_elems
    const Tensor2<T> srcpos;   // n_elems

    const Tensor2<T> grad_out;  // n_elems x channels
    Tensor5<T>
        grad_features;  // batch_size x n_views x channels x height x width

    MapToListNnBackward(const Tensor1<int> nbidx, const Tensor2<T> srcpos,
                        Tensor2<T> grad_out, Tensor5<T> grad_features)
        : nbidx(nbidx),
          srcpos(srcpos),
          grad_out(grad_out),
          grad_features(grad_features) {}

    CPU_GPU_FUNCTION void operator()(long idx) {
        // idx \in [0, n_elems]
        const long batch_size = grad_features.shape[0];
        const long channels = grad_features.shape[2];

        const int bidx = nbidx(idx) % batch_size;
        const int vidx = nbidx(idx) / batch_size;
        const int h = srcpos(idx, 0);
        const int w = srcpos(idx, 1);

        for (long c = 0; c < channels; ++c) {
            co_atomic_add(grad_features.ptridx(bidx, vidx, c, h, w),
                          grad_out(idx, c));
        }
    }
};
