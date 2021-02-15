#pragma once

#include <co_math.h>
#include <common.h>
#include <torch_common.h>
#include <torch_tensor.h>

template <typename T>
struct MapToListBlForward {
    const Tensor5<T>
        features;  // batch_size x n_views x channels x height x width
    const Tensor1<int> nbidx;  // n_elems
    const Tensor2<T> srcpos;   // n_elems x 2

    Tensor2<T> out;  // n_elems x channels

    MapToListBlForward(const Tensor5<T> features, const Tensor1<int> nbidx,
                       const Tensor2<T> srcpos, Tensor2<T> out)
        : features(features), nbidx(nbidx), srcpos(srcpos), out(out) {}

    CPU_GPU_FUNCTION void operator()(long idx) {
        // idx \in [0, n_elems]
        const long batch_size = features.shape[0];
        const long channels = features.shape[2];
        const long height = features.shape[3];
        const long width = features.shape[4];

        const int nbidx_ = nbidx(idx);
        const int bidx = nbidx_ % batch_size;
        const int vidx = nbidx_ / batch_size;

        // Pixel center is assumed to be at x+0.5, y+0.5
        const T h = srcpos(idx, 0);
        const T w = srcpos(idx, 1);

        T w1 = int(w - T(0.5)) + T(0.5);
        T h1 = int(h - T(0.5)) + T(0.5);
        T w2 = w1 + 1;
        T h2 = h1 + 1;

        const T tw1 = interp_triangle(w - w1);
        const T tw2 = interp_triangle(w2 - w);
        const T th1 = interp_triangle(h - h1);
        const T th2 = interp_triangle(h2 - h);

        w1 = co_min(co_max(w1, T(0)), T(width - 1));
        w2 = co_min(co_max(w2, T(0)), T(width - 1));
        h1 = co_min(co_max(h1, T(0)), T(height - 1));
        h2 = co_min(co_max(h2, T(0)), T(height - 1));

        for (long c = 0; c < channels; ++c) {
            out(idx, c) = features(bidx, vidx, c, h1, w1) * th1 * tw1 +
                          features(bidx, vidx, c, h1, w2) * th1 * tw2 +
                          features(bidx, vidx, c, h2, w1) * th2 * tw1 +
                          features(bidx, vidx, c, h2, w2) * th2 * tw2;
        }
    }
};

template <typename T>
struct MapToListBlBackward {
    const Tensor1<int> nbidx;  // n_elems
    const Tensor2<T> srcpos;   // n_elems

    const Tensor2<T> grad_out;  // n_elems x channels
    Tensor5<T>
        grad_features;  // batch_size x n_views x channels x height x width

    MapToListBlBackward(const Tensor1<int> nbidx, const Tensor2<T> srcpos,
                        const Tensor2<T> grad_out, Tensor5<T> grad_features)
        : nbidx(nbidx),
          srcpos(srcpos),
          grad_out(grad_out),
          grad_features(grad_features) {}

    CPU_GPU_FUNCTION void operator()(long idx) {
        // idx \in [0, n_elems]
        const long batch_size = grad_features.shape[0];
        const long channels = grad_features.shape[2];
        const long height = grad_features.shape[3];
        const long width = grad_features.shape[4];

        const int nbidx_ = nbidx(idx);
        const int bidx = nbidx_ % batch_size;
        const int vidx = nbidx_ / batch_size;

        const T h = srcpos(idx, 0);
        const T w = srcpos(idx, 1);

        T w1 = int(w - T(0.5)) + T(0.5);
        T h1 = int(h - T(0.5)) + T(0.5);
        T w2 = w1 + 1;
        T h2 = h1 + 1;

        const T tw1 = interp_triangle(w - w1);
        const T tw2 = interp_triangle(w2 - w);
        const T th1 = interp_triangle(h - h1);
        const T th2 = interp_triangle(h2 - h);

        w1 = co_min(co_max(w1, T(0)), T(width - 1));
        w2 = co_min(co_max(w2, T(0)), T(width - 1));
        h1 = co_min(co_max(h1, T(0)), T(height - 1));
        h2 = co_min(co_max(h2, T(0)), T(height - 1));

        for (long c = 0; c < channels; ++c) {
            const T go = grad_out(idx, c);
            co_atomic_add(grad_features.ptridx(bidx, vidx, c, h1, w1),
                          th1 * tw1 * go);
            co_atomic_add(grad_features.ptridx(bidx, vidx, c, h1, w2),
                          th1 * tw2 * go);
            co_atomic_add(grad_features.ptridx(bidx, vidx, c, h2, w1),
                          th2 * tw1 * go);
            co_atomic_add(grad_features.ptridx(bidx, vidx, c, h2, w2),
                          th2 * tw2 * go);
        }
    }
};
