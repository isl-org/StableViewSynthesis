#pragma once

#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT_CPU(x) CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_CUDA(x) \
    CHECK_CUDA(x);          \
    CHECK_CONTIGUOUS(x)

#define CHECK_DIM(x, correct_dim) AT_ASSERTM(x.dim() == correct_dim, #x " has invalid dimension")
#define CHECK_SHAPE_1(x, s0) \
    CHECK_DIM(x, 1);         \
    AT_ASSERTM(x.size(0) == s0, #x " invalid shape")
#define CHECK_SHAPE_2(x, s0, s1) \
    CHECK_DIM(x, 2);             \
    AT_ASSERTM(x.size(0) == s0 && x.size(1) == s1, #x " invalid shape")
#define CHECK_SHAPE_3(x, s0, s1, s2) \
    CHECK_DIM(x, 3);                 \
    AT_ASSERTM(x.size(0) == s0 && x.size(1) == s1 && x.size(2) == s2, #x " invalid shape")
#define CHECK_SHAPE_4(x, s0, s1, s2, s3)                                                 \
    CHECK_DIM(x, 4);                                                                     \
    AT_ASSERTM(x.size(0) == s0 && x.size(1) == s1 && x.size(2) == s2 && x.size(3) == s3, \
               #x " invalid shape")
#define CHECK_SHAPE_5(x, s0, s1, s2, s3, s4)                                               \
    CHECK_DIM(x, 5);                                                                       \
    AT_ASSERTM(x.size(0) == s0 && x.size(1) == s1 && x.size(2) == s2 && x.size(3) == s3 && \
                       x.size(4) == s4,                                                    \
               #x " invalid shape")
#define CHECK_SHAPE_6(x, s0, s1, s2, s3, s4, s5)                                           \
    CHECK_DIM(x, 6);                                                                       \
    AT_ASSERTM(x.size(0) == s0 && x.size(1) == s1 && x.size(2) == s2 && x.size(3) == s3 && \
                       x.size(4) == s4 && x.size(5) == s5,                                 \
               #x " invalid shape")

#define CHECK_INPUT_CPU_DIM(x, dim) \
    CHECK_INPUT_CPU(x);             \
    CHECK_DIM(x, dim)
#define CHECK_INPUT_CPU_SHAPE_1(x, s0) \
    CHECK_INPUT_CPU(x);                \
    CHECK_SHAPE_1(x, s0)
#define CHECK_INPUT_CPU_SHAPE_2(x, s0, s1) \
    CHECK_INPUT_CPU(x);                    \
    CHECK_SHAPE_2(x, s0, s1)
#define CHECK_INPUT_CPU_SHAPE_3(x, s0, s1, s2) \
    CHECK_INPUT_CPU(x);                        \
    CHECK_SHAPE_3(x, s0, s1, s2)
#define CHECK_INPUT_CPU_SHAPE_4(x, s0, s1, s2, s3) \
    CHECK_INPUT_CPU(x);                            \
    CHECK_SHAPE_4(x, s0, s1, s2, s3)
#define CHECK_INPUT_CPU_SHAPE_5(x, s0, s1, s2, s3, s4) \
    CHECK_INPUT_CPU(x);                                \
    CHECK_SHAPE_5(x, s0, s1, s2, s3, s4)
#define CHECK_INPUT_CPU_SHAPE_6(x, s0, s1, s2, s3, s4, s5) \
    CHECK_INPUT_CPU(x);                                    \
    CHECK_SHAPE_6(x, s0, s1, s2, s3, s4, s5)

#define CHECK_INPUT_CUDA_DIM(x, dim) \
    CHECK_INPUT_CUDA(x);             \
    CHECK_DIM(x, dim)
#define CHECK_INPUT_CUDA_SHAPE_1(x, s0) \
    CHECK_INPUT_CUDA(x);                \
    CHECK_SHAPE_1(x, s0)
#define CHECK_INPUT_CUDA_SHAPE_2(x, s0, s1) \
    CHECK_INPUT_CUDA(x);                    \
    CHECK_SHAPE_2(x, s0, s1)
#define CHECK_INPUT_CUDA_SHAPE_3(x, s0, s1, s2) \
    CHECK_INPUT_CUDA(x);                        \
    CHECK_SHAPE_3(x, s0, s1, s2)
#define CHECK_INPUT_CUDA_SHAPE_4(x, s0, s1, s2, s3) \
    CHECK_INPUT_CUDA(x);                            \
    CHECK_SHAPE_4(x, s0, s1, s2, s3)
#define CHECK_INPUT_CUDA_SHAPE_5(x, s0, s1, s2, s3, s4) \
    CHECK_INPUT_CUDA(x);                                \
    CHECK_SHAPE_5(x, s0, s1, s2, s3, s4)
#define CHECK_INPUT_CUDA_SHAPE_6(x, s0, s1, s2, s3, s4, s5) \
    CHECK_INPUT_CUDA(x);                                    \
    CHECK_SHAPE_6(x, s0, s1, s2, s3, s4, s5)
