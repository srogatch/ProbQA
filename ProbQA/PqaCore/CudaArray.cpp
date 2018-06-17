// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CudaArray.h"
#include "../PqaCore/CudaMacros.h"

namespace ProbQA {

CudaArray::CudaArray(const int64_t nBytes) {
  CUDA_MUST(cudaMalloc(&_d_p, nBytes));
}

void CudaArray::Destroy(void* d_p) {
  CUDA_MUST(cudaFree(d_p));
}

} // namespace ProbQA
