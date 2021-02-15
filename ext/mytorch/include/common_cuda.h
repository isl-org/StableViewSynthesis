/*
This file contains code snippets from 
v1.6.0.tar.gz/pytorch-b31f58de6fa8bbda5353b3c77d9be4914399724d/caffe2/core/common_gpu.cc
To this end, we include the following copyright notice:

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <cublas_v2.h>
#include <stdio.h>

#define DEBUG 0
#define CUDA_DEBUG_DEVICE_SYNC 0

// cuda check for cudaMalloc and so on
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    if(CUDA_DEBUG_DEVICE_SYNC) { cudaDeviceSynchronize(); } \
    cudaError_t error = condition; \
    if(error != cudaSuccess) { \
      printf("%s in %s at %d\n", cudaGetErrorString(error), __FILE__, __LINE__); \
      exit(-1); \
    } \
  } while (0)

/// Get error string for error code.
/// @param error
inline const char* cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "Unknown cublas status";
}

#define CUBLAS_CHECK(condition) \
  do { \
    if(CUDA_DEBUG_DEVICE_SYNC) { cudaDeviceSynchronize(); } \
    cublasStatus_t status = condition; \
    if(status != CUBLAS_STATUS_SUCCESS) { \
      printf("%s in %s at %d\n", cublasGetErrorString(status), __FILE__, __LINE__); \
      exit(-1); \
    } \
  } while (0)

// check if there is a error after kernel execution
#define CUDA_POST_KERNEL_CHECK \
  CUDA_CHECK(cudaPeekAtLastError()); \
  CUDA_CHECK(cudaGetLastError());

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N, const int N_THREADS=CUDA_NUM_THREADS) {
  return (N + N_THREADS - 1) / N_THREADS;
}

template<typename T>
T* device_malloc(long N) {
  T* dptr;
  CUDA_CHECK(cudaMalloc(&dptr, N * sizeof(T)));
  if(DEBUG) { printf("[DEBUG] device_malloc %p, %ld\n", dptr, N); }
  return dptr;
}

template<typename T>
void device_free(T* dptr) {
  if(DEBUG) { printf("[DEBUG] device_free %p\n", dptr); }
  CUDA_CHECK(cudaFree(dptr));
}

template<typename T>
void host_to_device(const T* hptr, T* dptr, long N) {
  if(DEBUG) { printf("[DEBUG] host_to_device %p => %p, %ld\n", hptr, dptr, N); }
  CUDA_CHECK(cudaMemcpy(dptr, hptr, N * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
T* host_to_device_malloc(const T* hptr, long N) {
  T* dptr = device_malloc<T>(N);
  host_to_device(hptr, dptr, N);
  return dptr;
}

template<typename T>
void device_to_host(const T* dptr, T* hptr, long N) {
  if(DEBUG) { printf("[DEBUG] device_to_host %p => %p, %ld\n", dptr, hptr, N); }
  CUDA_CHECK(cudaMemcpy(hptr, dptr, N * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
T* device_to_host_malloc(const T* dptr, long N) {
  T* hptr = new T[N];
  device_to_host(dptr, hptr, N);
  return hptr;
}

template<typename T>
void device_to_device(const T* dptr, T* hptr, long N) {
  if(DEBUG) { printf("[DEBUG] device_to_device %p => %p, %ld\n", dptr, hptr, N); }
  CUDA_CHECK(cudaMemcpy(hptr, dptr, N * sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename FunctorT>
__global__ void iterate_kernel(FunctorT functor, int N) {
  CUDA_KERNEL_LOOP(idx, N) {
    functor(idx);
  }
}

template <typename FunctorT>
void iterate_cuda(FunctorT functor, int N, int N_THREADS=CUDA_NUM_THREADS) {
  iterate_kernel<<<GET_BLOCKS(N, N_THREADS), N_THREADS>>>(functor, N);
  CUDA_POST_KERNEL_CHECK;
}

