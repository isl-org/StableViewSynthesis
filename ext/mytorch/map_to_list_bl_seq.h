
#pragma once

#include <co_math.h>
#include <common.h>
#include <torch_common.h>
#include <torch_tensor.h>

template <typename T>
struct MapToListBlSeqForward {
    const Tensor3<T> features;  // channels x height x width
    const Tensor1<int> tgtidx;  // n_elems_feat
    const Tensor2<T> srcpos;    // n_elems_feat x 2

    Tensor2<T> out;  // n_elems x channels

    MapToListBlSeqForward(const Tensor3<T> features, const Tensor1<int> tgtidx,
                          const Tensor2<T> srcpos, Tensor2<T> out)
        : features(features), tgtidx(tgtidx), srcpos(srcpos), out(out) {}

    CPU_GPU_FUNCTION void operator()(long idx) {
        // idx \in [0, n_elems_feat]

        const long channels = features.shape[0];
        const long height = features.shape[1];
        const long width = features.shape[2];

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

        int out_idx = tgtidx[idx];
        // if (out_idx > out.shape[0]) {
        //     printf("out_idx %d > out.shape[0] %ld\n", out_idx, out.shape[0]);
        // }
        for (long c = 0; c < channels; ++c) {
            out(out_idx, c) = features(c, h1, w1) * th1 * tw1 +
                              features(c, h1, w2) * th1 * tw2 +
                              features(c, h2, w1) * th2 * tw1 +
                              features(c, h2, w2) * th2 * tw2;
        }
    }
};
