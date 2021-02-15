#pragma once

#include <cmath>

#include "co_types.h"


template <typename T, int N=3>
CPU_GPU_FUNCTION
inline void co_vec_fill(T* v, const T fill) {
  for(int idx = 0; idx < N; ++idx) {
    v[idx] = fill;
  }
}

template <>
CPU_GPU_FUNCTION
inline void co_vec_fill<float, 3>(float* v, const float fill) {
  v[0] = fill;
  v[1] = fill;
  v[2] = fill;
}

template <typename T, T fill, int N=3>
CPU_GPU_FUNCTION
inline void co_vec_fill(T* v) {
  for(int idx = 0; idx < N; ++idx) {
    v[idx] = fill;
  }
}

template <typename T, int N=3>
CPU_GPU_FUNCTION
inline void co_vec_add(const T* in1, const T* in2, T* out) {
  for(int idx = 0; idx < N; ++idx) {
    out[idx] = in1[idx] + in2[idx];
  }
}

template <>
CPU_GPU_FUNCTION
inline void co_vec_add<float, 3>(const float* in1, const float* in2, float* out) {
  out[0] = in1[0] + in2[0];
  out[1] = in1[1] + in2[1];
  out[2] = in1[2] + in2[2];
}

template <typename T, int N=3>
CPU_GPU_FUNCTION
inline void co_vec_add(const T lam1, const T* in1, const T lam2, const T* in2, T* out) {
  for(int idx = 0; idx < N; ++idx) {
    out[idx] = lam1 * in1[idx] + lam2 * in2[idx];
  }
}

template <>
CPU_GPU_FUNCTION
inline void co_vec_add<float, 3>(const float lam1, const float* in1, const float lam2, const float* in2, float* out) {
  out[0] = lam1 * in1[0] + lam2 * in2[0];
  out[1] = lam1 * in1[1] + lam2 * in2[1];
  out[2] = lam1 * in1[2] + lam2 * in2[2];
}

template <typename T, int N=3>
CPU_GPU_FUNCTION
inline void co_vec_sub(const T* in1, const T* in2, T* out) {
  for(int idx = 0; idx < N; ++idx) {
    out[idx] = in1[idx] - in2[idx];
  }
}

template <>
CPU_GPU_FUNCTION
inline void co_vec_sub<float, 3>(const float* in1, const float* in2, float* out) {
  out[0] = in1[0] - in2[0];
  out[1] = in1[1] - in2[1];
  out[2] = in1[2] - in2[2];
}

template <typename T, int N=3>
CPU_GPU_FUNCTION
inline void co_vec_add_scalar(const T* in, const T lam, T* out) {
  for(int idx = 0; idx < N; ++idx) {
    out[idx] = in[idx] + lam;
  }
}

template <>
CPU_GPU_FUNCTION
inline void co_vec_add_scalar<float, 3>(const float* in, const float lam, float* out) {
  out[0] = in[0] + lam;
  out[1] = in[1] + lam;
  out[2] = in[2] + lam;
}

template <typename T, int N=3>
CPU_GPU_FUNCTION
inline void co_vec_mul_scalar(const T* in, const T lam, T* out) {
  for(int idx = 0; idx < N; ++idx) {
    out[idx] = in[idx] * lam;
  }
}

template <>
CPU_GPU_FUNCTION
inline void co_vec_mul_scalar<float, 3>(const float* in, const float lam, float* out) {
  out[0] = in[0] * lam;
  out[1] = in[1] * lam;
  out[2] = in[2] * lam;
}

template <typename T, int N=3>
CPU_GPU_FUNCTION
inline void co_vec_div_scalar(const T* in, const T lam, T* out) {
  for(int idx = 0; idx < N; ++idx) {
    out[idx] = in[idx] / lam;
  }
}

template <>
CPU_GPU_FUNCTION
inline void co_vec_div_scalar<float, 3>(const float* in, const float lam, float* out) {
  out[0] = in[0] / lam;
  out[1] = in[1] / lam;
  out[2] = in[2] / lam;
}

template <typename T>
CPU_GPU_FUNCTION
inline void co_mat_dot_vec3(const T* M, const T* v, T* w) {
  w[0] = M[0] * v[0] + M[1] * v[1] + M[2] * v[2];
  w[1] = M[3] * v[0] + M[4] * v[1] + M[5] * v[2];
  w[2] = M[6] * v[0] + M[7] * v[1] + M[8] * v[2];
}

template <typename T>
CPU_GPU_FUNCTION
inline void co_matT_dot_vec3(const T* M, const T* v, T* w) {
  w[0] = M[0] * v[0] + M[3] * v[1] + M[6] * v[2];
  w[1] = M[1] * v[0] + M[4] * v[1] + M[7] * v[2];
  w[2] = M[2] * v[0] + M[5] * v[1] + M[8] * v[2];
}

template <typename T>
CPU_GPU_FUNCTION
inline void co_mat34_dot_vec3(const T* M, const T* v, T* w) {
  w[0] = M[0] * v[0] + M[1] * v[1] + M[2] * v[2] + M[3] * v[3];
  w[1] = M[4] * v[0] + M[5] * v[1] + M[6] * v[2] + M[7] * v[3];
  w[2] = M[8] * v[0] + M[9] * v[1] + M[10] * v[2] + M[11] * v[3];
}

template <typename T>
CPU_GPU_FUNCTION
inline void co_mat_dot_mat3(const T* A, const T* B, T* C) {
  C[0] = A[0] * B[0] + A[1] * B[3] + A[2] * B[6];
  C[1] = A[0] * B[1] + A[1] * B[4] + A[2] * B[7];
  C[2] = A[0] * B[2] + A[1] * B[5] + A[2] * B[8];

  C[3] = A[3] * B[0] + A[4] * B[3] + A[5] * B[6];
  C[4] = A[3] * B[1] + A[4] * B[4] + A[5] * B[7];
  C[5] = A[3] * B[2] + A[4] * B[5] + A[5] * B[8];

  C[6] = A[6] * B[0] + A[7] * B[3] + A[8] * B[6];
  C[7] = A[6] * B[1] + A[7] * B[4] + A[8] * B[7];
  C[8] = A[6] * B[2] + A[7] * B[5] + A[8] * B[8];
}

template <typename T>
CPU_GPU_FUNCTION
inline void co_matT_dot_mat3(const T* A, const T* B, T* C) {
  C[0] = A[0] * B[0] + A[3] * B[3] + A[6] * B[6];
  C[1] = A[0] * B[1] + A[3] * B[4] + A[6] * B[7];
  C[2] = A[0] * B[2] + A[3] * B[5] + A[6] * B[8];

  C[3] = A[1] * B[0] + A[4] * B[3] + A[7] * B[6];
  C[4] = A[1] * B[1] + A[4] * B[4] + A[7] * B[7];
  C[5] = A[1] * B[2] + A[4] * B[5] + A[7] * B[8];

  C[6] = A[2] * B[0] + A[5] * B[3] + A[8] * B[6];
  C[7] = A[2] * B[1] + A[5] * B[4] + A[8] * B[7];
  C[8] = A[2] * B[2] + A[5] * B[5] + A[8] * B[8];
}

template <typename T>
CPU_GPU_FUNCTION
inline void co_transform_vec3(const T* R, const T* t, const T* x, T* y) {
  y[0] = R[0] * x[0] + R[1] * x[1] + R[2] * x[2] + t[0];
  y[1] = R[3] * x[0] + R[4] * x[1] + R[5] * x[2] + t[1];
  y[2] = R[6] * x[0] + R[7] * x[1] + R[8] * x[2] + t[2];
}

template <typename T>
CPU_GPU_FUNCTION
inline void co_transform_inv_vec3(const T* R, const T* t, const T* x, T* y) {
  y[0] = R[0] * (x[0]-t[0]) + R[3] * (x[1]-t[1]) + R[6] * (x[2]-t[2]);
  y[1] = R[1] * (x[0]-t[0]) + R[4] * (x[1]-t[1]) + R[7] * (x[2]-t[2]);
  y[2] = R[2] * (x[0]-t[0]) + R[5] * (x[1]-t[1]) + R[8] * (x[2]-t[2]);
}

template <typename T, int N=3>
CPU_GPU_FUNCTION
inline T co_vec_dot(const T* in1, const T* in2) {
  T out = T(0);
  for(int idx = 0; idx < N; ++idx) {
    out += in1[idx] * in2[idx];
  }
  return out;
}

template <>
CPU_GPU_FUNCTION
inline float co_vec_dot<float, 3>(const float* in1, const float* in2) {
  return in1[0] * in2[0] + in1[1] * in2[1] + in1[2] * in2[2];
}

template <typename T>
CPU_GPU_FUNCTION
inline void co_vec_cross3(const T* u, const T* v, T* out) {
  out[0] = u[1] * v[2] - u[2] * v[1];
  out[1] = u[2] * v[0] - u[0] * v[2];
  out[2] = u[0] * v[1] - u[1] * v[0];
}

template <typename T, int N=3>
CPU_GPU_FUNCTION
inline T co_vec_norm(const T* u) {
  T norm = T(0);
  for(int idx = 0; idx < N; ++idx) {
    norm += u[idx] * u[idx];
  }
  return std::sqrt(norm);
}

template <>
CPU_GPU_FUNCTION
inline float co_vec_norm<float, 3>(const float* u) {
  return std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
}

template <typename T, int N=3>
CPU_GPU_FUNCTION
inline void co_vec_normalize(const T* u, T* v) {
  T denom = co_vec_norm(u);
  co_vec_div_scalar(u, denom, v);
}

template <>
CPU_GPU_FUNCTION
inline void co_vec_normalize<float, 3>(const float* u, float* v) {
  co_vec_div_scalar(u, co_vec_norm(u), v);
}

template <typename T>
CPU_GPU_FUNCTION
bool co_solve_quadratic(const T& a, const T& b, const T& c, T& x0, T& x1) {
  T discr = b * b - 4 * a * c;
  if(discr < 0) {
    return false;
  }
  else if(discr == 0) {
    x0 = -0.5 * b / a;
    x1 = x0;
  }
  else {
    T q = (b > 0) ?
        -0.5 * (b + std::sqrt(discr)) :
        -0.5 * (b - std::sqrt(discr));
    x0 = q / a;
    x1 = c / q;
  }

  return true;
}
