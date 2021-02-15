#pragma once

#include "co_types.h"

template <typename T, long dim>
struct Tensor {
  T* data;
  const long shape[dim];

  Tensor() : data(nullptr) {}
  Tensor(T* data) : data(data), shape{0} {}
  Tensor(long s0, T* data) : data(data), shape{s0} {}
  Tensor(long s0, long s1, T* data) : data(data), shape{s0, s1} {}
  Tensor(long s0, long s1, long s2, T* data) : data(data), shape{s0, s1, s2} {}
  Tensor(long s0, long s1, long s2, long s3, T* data) : data(data), shape{s0, s1, s2, s3} {}
  Tensor(long s0, long s1, long s2, long s3, long s4, T* data) : data(data), shape{s0, s1, s2, s3, s4} {}
  Tensor(long s0, long s1, long s2, long s3, long s4, long s5, T* data) : data(data), shape{s0, s1, s2, s3, s4, s5} {}
  Tensor(long s0, long s1, long s2, long s3, long s4, long s5, long s6, T* data) : data(data), shape{s0, s1, s2, s3, s4, s5, s6} {}
  Tensor(long s0, long s1, long s2, long s3, long s4, long s5, long s6, long s7, T* data) : data(data), shape{s0, s1, s2, s3, s4, s5, s6, s7} {}

  CPU_GPU_FUNCTION
  inline const T& operator[](long idx) const { return data[idx]; }
  CPU_GPU_FUNCTION
  inline T& operator[](long idx) { return data[idx]; }
  CPU_GPU_FUNCTION
  inline T* ptr(long offset) const { return data + offset; }

  CPU_GPU_FUNCTION
  inline bool inbounds(long idx, long axis) const {
    return ((idx >= 0) && (idx < this->shape[axis]));
  }
};


template <typename T>
struct Tensor1 : public Tensor<T,1> {
  Tensor1(long s0, T* data) : Tensor<T,1>(s0, data) {}

  CPU_GPU_FUNCTION
  inline long idx(long idx0) const {
    return idx0;
  }
  CPU_GPU_FUNCTION
  inline const T& operator()(long idx0) const { return this->data[idx(idx0)]; }
  CPU_GPU_FUNCTION
  inline T& operator()(long idx0) { return this->data[idx(idx0)]; }
  CPU_GPU_FUNCTION
  inline T* ptridx(long idx0) const { return this->ptr(idx(idx0)); }
};

template <typename T>
struct Tensor2 : public Tensor<T,2> {
  Tensor2(long s0, long s1, T* data) : Tensor<T,2>(s0, s1, data) {}

  CPU_GPU_FUNCTION
  inline long idx(long idx0, long idx1) const {
    return idx0 * this->shape[1] + idx1;
  }
  CPU_GPU_FUNCTION
  inline const T& operator()(long idx0, long idx1) const { return this->data[idx(idx0,idx1)]; }
  CPU_GPU_FUNCTION
  inline T& operator()(long idx0, long idx1) { return this->data[idx(idx0,idx1)]; }
  CPU_GPU_FUNCTION
  inline T* ptridx(long idx0, long idx1) const { return this->ptr(idx(idx0,idx1)); }
};

template <typename T>
struct Tensor3 : public Tensor<T,3> {
  Tensor3(long s0, long s1, long s2, T* data) : Tensor<T,3>(s0, s1, s2, data) {}

  CPU_GPU_FUNCTION
  inline long idx(long idx0, long idx1, long idx2) const {
    return (idx0 * this->shape[1] + idx1) * this->shape[2] + idx2;
  }
  CPU_GPU_FUNCTION
  inline const T& operator()(long idx0, long idx1, long idx2) const { return this->data[idx(idx0,idx1,idx2)]; }
  CPU_GPU_FUNCTION
  inline T& operator()(long idx0, long idx1, long idx2) { return this->data[idx(idx0,idx1,idx2)]; }
  CPU_GPU_FUNCTION
  inline T* ptridx(long idx0, long idx1, long idx2) const { return this->ptr(idx(idx0,idx1,idx2)); }
};

template <typename T>
struct Tensor4 : public Tensor<T,4> {
  Tensor4(long s0, long s1, long s2, long s3, T* data) : Tensor<T,4>(s0, s1, s2, s3, data) {}

  CPU_GPU_FUNCTION
  inline long idx(long idx0, long idx1, long idx2, long idx3) const {
    return ((idx0 * this->shape[1] + idx1) * this->shape[2] + idx2) * this->shape[3] + idx3;
  }
  CPU_GPU_FUNCTION
  inline const T& operator()(long idx0, long idx1, long idx2, long idx3) const { return this->data[idx(idx0,idx1,idx2,idx3)]; }
  CPU_GPU_FUNCTION
  inline T& operator()(long idx0, long idx1, long idx2, long idx3) { return this->data[idx(idx0,idx1,idx2,idx3)]; }
  CPU_GPU_FUNCTION
  inline T* ptridx(long idx0, long idx1, long idx2, long idx3) const { return this->ptr(idx(idx0,idx1,idx2,idx3)); }
};

template <typename T>
struct Tensor5 : public Tensor<T,5> {
  Tensor5(long s0, long s1, long s2, long s3, long s4, T* data) : Tensor<T,5>(s0, s1, s2, s3, s4, data) {}

  CPU_GPU_FUNCTION
  inline long idx(long idx0, long idx1, long idx2, long idx3, long idx4) const {
    return (((idx0 * this->shape[1] + idx1) * this->shape[2] + idx2) * this->shape[3] + idx3) * this->shape[4] + idx4;
  }
  CPU_GPU_FUNCTION
  inline const T& operator()(long idx0, long idx1, long idx2, long idx3, long idx4) const { return this->data[idx(idx0,idx1,idx2,idx3,idx4)]; }
  CPU_GPU_FUNCTION
  inline T& operator()(long idx0, long idx1, long idx2, long idx3, long idx4) { return this->data[idx(idx0,idx1,idx2,idx3,idx4)]; }
  CPU_GPU_FUNCTION
  inline T* ptridx(long idx0, long idx1, long idx2, long idx3, long idx4) const { return this->ptr(idx(idx0,idx1,idx2,idx3,idx4)); }
};

template <typename T>
struct Tensor6 : public Tensor<T,6> {
  Tensor6(long s0, long s1, long s2, long s3, long s4, long s5, T* data) : Tensor<T,6>(s0, s1, s2, s3, s4, s5, data) {}

  CPU_GPU_FUNCTION
  inline long idx(long idx0, long idx1, long idx2, long idx3, long idx4, long idx5) const {
    return ((((idx0 * this->shape[1] + idx1) * this->shape[2] + idx2) * this->shape[3] + idx3) * this->shape[4] + idx4) * this->shape[5] + idx5;
  }
  CPU_GPU_FUNCTION
  inline const T& operator()(long idx0, long idx1, long idx2, long idx3, long idx4, long idx5) const { return this->data[idx(idx0,idx1,idx2,idx3,idx4,idx5)]; }
  CPU_GPU_FUNCTION
  inline T& operator()(long idx0, long idx1, long idx2, long idx3, long idx4, long idx5) { return this->data[idx(idx0,idx1,idx2,idx3,idx4,idx5)]; }
  CPU_GPU_FUNCTION
  inline T* ptridx(long idx0, long idx1, long idx2, long idx3, long idx4, long idx5) const { return this->ptr(idx(idx0,idx1,idx2,idx3,idx4,idx5)); }
};

template <typename T>
struct Tensor7 : public Tensor<T,7> {
  Tensor7(long s0, long s1, long s2, long s3, long s4, long s5, long s6, T* data) : Tensor<T,7>(s0, s1, s2, s3, s4, s5, s6, data) {}

  CPU_GPU_FUNCTION
  inline long idx(long idx0, long idx1, long idx2, long idx3, long idx4, long idx5, long idx6) const {
    return (((((idx0 * this->shape[1] + idx1) * this->shape[2] + idx2) * this->shape[3] + idx3) * this->shape[4] + idx4) * this->shape[5] + idx5) * this->shape[6] + idx6;
  }
  CPU_GPU_FUNCTION
  inline const T& operator()(long idx0, long idx1, long idx2, long idx3, long idx4, long idx5, long idx6) const { return this->data[idx(idx0,idx1,idx2,idx3,idx4,idx5,idx6)]; }
  CPU_GPU_FUNCTION
  inline T& operator()(long idx0, long idx1, long idx2, long idx3, long idx4, long idx5, long idx6) { return this->data[idx(idx0,idx1,idx2,idx3,idx4,idx5,idx6)]; }
  CPU_GPU_FUNCTION
  inline T* ptridx(long idx0, long idx1, long idx2, long idx3, long idx4, long idx5, long idx6) const { return this->ptr(idx(idx0,idx1,idx2,idx3,idx4,idx5,idx6)); }
};

template <typename T>
struct Tensor8 : public Tensor<T,8> {
  Tensor8(long s0, long s1, long s2, long s3, long s4, long s5, long s6, long s7, T* data) : Tensor<T,8>(s0, s1, s2, s3, s4, s5, s6, s7, data) {}

  CPU_GPU_FUNCTION
  inline long idx(long idx0, long idx1, long idx2, long idx3, long idx4, long idx5, long idx6, long idx7) const {
    return ((((((idx0 * this->shape[1] + idx1) * this->shape[2] + idx2) * this->shape[3] + idx3) * this->shape[4] + idx4) * this->shape[5] + idx5) * this->shape[6] + idx6) * this->shape[7] + idx7;
  }
  CPU_GPU_FUNCTION
  inline const T& operator()(long idx0, long idx1, long idx2, long idx3, long idx4, long idx5, long idx6) const { return this->data[idx(idx0,idx1,idx2,idx3,idx4,idx5,idx6)]; }
  CPU_GPU_FUNCTION
  inline T& operator()(long idx0, long idx1, long idx2, long idx3, long idx4, long idx5, long idx6, long idx7) { return this->data[idx(idx0,idx1,idx2,idx3,idx4,idx5,idx6,idx7)]; }
  CPU_GPU_FUNCTION
  inline T* ptridx(long idx0, long idx1, long idx2, long idx3, long idx4, long idx5, long idx6, long idx7) const { return this->ptr(idx(idx0,idx1,idx2,idx3,idx4,idx5,idx6,idx7)); }
};
