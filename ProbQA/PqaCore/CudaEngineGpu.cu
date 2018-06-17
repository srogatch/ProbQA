// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include <cstdint>
#include <algorithm>
#include "../PqaCore/CudaEngineGpu.cuh"

namespace ProbQA {

template<typename taNumber> __global__ void InitStatistics(const InitStatisticsKernel<taNumber> isk) {
  const int32_t nThreads = gridDim.x * blockDim.x;
  int64_t iInstance = threadIdx.x + blockIdx.x * blockDim.x;
  while (iInstance < isk._nSAItems) {
    isk._psA[iInstance] = isk._initSqr;
    iInstance += nThreads;
  }

  iInstance = threadIdx.x + blockIdx.x * blockDim.x;
  while (iInstance < isk._nMDItems) {
    isk._pmD[iInstance] = isk._initMD;
    iInstance += nThreads;
  }

  iInstance = threadIdx.x + blockIdx.x * blockDim.x;
  while (iInstance < isk._nVBItems) {
    isk._pvB[iInstance] = isk._init1;
    iInstance += nThreads;
  }
}

template<typename taNumber> void InitStatisticsKernel<taNumber>::Run(const KernelLaunchContext& klc,
  cudaStream_t stream)
{
  const int64_t nBlocks = klc.GetBlockCount(_nSAItems);
  InitStatistics<taNumber> <<<nBlocks, klc.GetBlockSize(), /* no shared memory */ 0, stream>>> (*this);
}

} // namespace ProbQA
