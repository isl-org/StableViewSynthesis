#pragma once

#if defined(_OPENMP)
#include <omp.h>
#endif

template <typename FunctorT>
void iterate_cpu(FunctorT functor, int N) {
  for(int idx = 0; idx < N; ++idx) {
    functor(idx);
  }
}

template <typename FunctorT>
void iterate_omp_cpu(FunctorT functor, int N, int n_threads) {
#if defined(_OPENMP)
  omp_set_num_threads(n_threads);
  #pragma omp parallel for
#endif
  for(int idx = 0; idx < N; ++idx) {
    functor(idx);
  }
}
