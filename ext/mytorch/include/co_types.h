#pragma once

#ifdef __CUDA_ARCH__
#define CPU_GPU_FUNCTION __host__ __device__
#else
#define CPU_GPU_FUNCTION
#endif


#define DATA_FORMAT_NCHW 0
#define DATA_FORMAT_NWHC 1
