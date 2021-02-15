#pragma once

#include <algorithm>
#include <cmath>
#include "co_types.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

#define DISABLE_COPY_AND_ASSIGN(classname) \
   private:                                \
    classname(const classname&) = delete;  \
    classname& operator=(const classname&) = delete;

template <typename T>
struct FillFunctor {
    T* arr;
    const T val;

    FillFunctor(T* arr, const T val) : arr(arr), val(val) {}
    CPU_GPU_FUNCTION void operator()(const int idx) { arr[idx] = val; }
};

CPU_GPU_FUNCTION
inline void co_syncthreads() {
#ifdef __CUDA_ARCH__
    __syncthreads();
#else
#if defined(_OPENMP)
#pragma omp barrier
#endif
#endif
}

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(
            address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN
        // != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif
#endif

template <typename T>
CPU_GPU_FUNCTION inline void co_atomic_add(T* addr, T val) {
#ifdef __CUDA_ARCH__
    atomicAdd(addr, val);
#else
#if defined(_OPENMP)
#pragma omp atomic
#endif
    *addr += val;
#endif
}

CPU_GPU_FUNCTION
inline void co_atomic_min(float* addr, float value) {
#ifdef __CUDA_ARCH__
    if (value >= 0) {
        __int_as_float(atomicMin((int*)addr, __float_as_int(value)));
    } else {
        __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));
    }
#else
#if defined(_OPENMP)
#pragma omp atomic
#endif
    *addr = std::min(*addr, value);
#endif
}

CPU_GPU_FUNCTION
inline void co_atomic_min(double* addr, double value) {
#ifdef __CUDA_ARCH__
    unsigned long long ret = __double_as_longlong(*addr);
    while (value < __longlong_as_double(ret)) {
        unsigned long long old = ret;
        if ((ret = atomicCAS((unsigned long long*)addr, old,
                             __double_as_longlong(value))) == old)
            break;
    }
#else
#if defined(_OPENMP)
#pragma omp atomic
#endif
    *addr = std::min(*addr, value);
#endif
}

CPU_GPU_FUNCTION
inline void co_atomic_max(float* addr, float value) {
#ifdef __CUDA_ARCH__
    if (value >= 0) {
        __int_as_float(atomicMax((int*)addr, __float_as_int(value)));
    } else {
        __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
    }
#else
#if defined(_OPENMP)
#pragma omp atomic
#endif
    *addr = std::max(*addr, value);
#endif
}

CPU_GPU_FUNCTION
inline void co_atomic_max(double* addr, double value) {
#ifdef __CUDA_ARCH__
    unsigned long long ret = __double_as_longlong(*addr);
    while (value > __longlong_as_double(ret)) {
        unsigned long long old = ret;
        if ((ret = atomicCAS((unsigned long long*)addr, old,
                             __double_as_longlong(value))) == old)
            break;
    }
#else
#if defined(_OPENMP)
#pragma omp atomic
#endif
    *addr = std::max(*addr, value);
#endif
}

template <typename T>
CPU_GPU_FUNCTION inline T co_exp(const T& a) {
#ifdef __CUDA_ARCH__
    return exp(a);
#else
    return std::exp(a);
#endif
}

template <typename T>
CPU_GPU_FUNCTION inline T co_abs(const T& a) {
#ifdef __CUDA_ARCH__
    return abs(a);
#else
    return std::abs(a);
#endif
}

// template <>
// CPU_GPU_FUNCTION
// inline float co_abs<float>(const float& a) {
// #ifdef __CUDA_ARCH__
//   return fabsf(a);
// #else
//   return std::abs(a);
// #endif
// }

template <typename T>
CPU_GPU_FUNCTION inline T co_min(const T& a, const T& b) {
#ifdef __CUDA_ARCH__
    return min(a, b);
#else
    return std::min(a, b);
#endif
}

// template <>
// CPU_GPU_FUNCTION
// inline float co_min<float>(const float& a, const float& b) {
// #ifdef __CUDA_ARCH__
//   return fminf(a, b);
// #else
//   return std::min(a, b);
// #endif
// }

template <typename T>
CPU_GPU_FUNCTION inline T co_max(const T& a, const T& b) {
#ifdef __CUDA_ARCH__
    return max(a, b);
#else
    return std::max(a, b);
#endif
}

// template <>
// CPU_GPU_FUNCTION
// inline float co_max<float>(const float& a, const float& b) {
// #ifdef __CUDA_ARCH__
//   return fmaxf(a, b);
// #else
//   return std::max(a, b);
// #endif
// }

template <typename T>
CPU_GPU_FUNCTION inline T co_round(const T& a) {
#ifdef __CUDA_ARCH__
    return round(a);
#else
    return round(a);
#endif
}

// template <>
// CPU_GPU_FUNCTION
// inline float co_round(const float& a) {
// #ifdef __CUDA_ARCH__
//   return roundf(a);
// #else
//   return round(a);
// #endif
// }

template <typename T>
CPU_GPU_FUNCTION inline T co_floor(const T& a) {
#ifdef __CUDA_ARCH__
    return floor(a);
#else
    return std::floor(a);
#endif
}

// template <>
// CPU_GPU_FUNCTION
// inline float co_floor(const float& a) {
// #ifdef __CUDA_ARCH__
//   return floorf(a);
// #else
//   return std::floor(a);
// #endif
// }

template <typename T>
CPU_GPU_FUNCTION inline T interp_triangle(T x) {
    // return (-0.5 <= x) && (x < 0.5);
    return (x + 1) * ((-1 <= x) && (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1));
}

template <typename T>
CPU_GPU_FUNCTION inline T interp_triangle_bwd(T x) {
    // return 0;
    return (1) * ((-1 <= x) && (x < 0)) + (-1) * ((0 <= x) & (x <= 1));
}

