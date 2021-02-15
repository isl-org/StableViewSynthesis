#pragma once

#include <torch/extension.h>
#include "tensor.h"

template <typename T>
inline Tensor1<T> torch2co1(at::Tensor t) {
    return Tensor1<T>(t.size(0), t.data_ptr<T>());
}

template <typename T>
inline Tensor2<T> torch2co2(at::Tensor t) {
    return Tensor2<T>(t.size(0), t.size(1), t.data_ptr<T>());
}

template <typename T>
inline Tensor3<T> torch2co3(at::Tensor t) {
    return Tensor3<T>(t.size(0), t.size(1), t.size(2), t.data_ptr<T>());
}

template <typename T>
inline Tensor4<T> torch2co4(at::Tensor t) {
    return Tensor4<T>(t.size(0), t.size(1), t.size(2), t.size(3), t.data_ptr<T>());
}

template <typename T>
inline Tensor5<T> torch2co5(at::Tensor t) {
    return Tensor5<T>(t.size(0), t.size(1), t.size(2), t.size(3), t.size(4), t.data_ptr<T>());
}

template <typename T>
inline Tensor6<T> torch2co6(at::Tensor t) {
    return Tensor6<T>(t.size(0), t.size(1), t.size(2), t.size(3), t.size(4), t.size(5),
                      t.data_ptr<T>());
}

template <typename T>
inline Tensor7<T> torch2co7(at::Tensor t) {
    return Tensor7<T>(t.size(0), t.size(1), t.size(2), t.size(3), t.size(4), t.size(5), t.size(6), t.data_ptr<T>());
}

template <typename T>
inline Tensor8<T> torch2co8(at::Tensor t) {
    return Tensor8<T>(t.size(0), t.size(1), t.size(2), t.size(3), t.size(4), t.size(5), t.size(6), t.size(7), t.data_ptr<T>());
}
