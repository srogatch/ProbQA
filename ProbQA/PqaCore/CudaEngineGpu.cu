// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include <cstdint>
#include <algorithm>
#include <cuda_runtime.h>
#include "../PqaCore/CudaEngineGpu.cuh"

namespace ProbQA {

template<typename taNumber> class DevAccumulator {
  taNumber _sum, _corr;
public:
  __device__ DevAccumulator& Init(const taNumber value) {
    _sum = value;
    _corr = 0;
    return *this;
  }

  __device__ DevAccumulator& Add(const taNumber value) {
    const taNumber y = value - _corr;
    const taNumber t = _sum + y;
    _corr = (t - _sum) - y;
    _sum = t;
    return *this;
  }

  __device__ DevAccumulator& Add(const DevAccumulator<taNumber>& fellow) {
    return Add(fellow._sum).Add(fellow._corr);
  }

  __device__ taNumber Get() {
    return _sum;
  }
};


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
  const uint32_t nBlocks = klc.GetBlockCount(_nSAItems);
  InitStatistics<taNumber> <<<nBlocks, klc.DefaultBlockSize(), /* no shared memory */ 0, stream>>> (*this);
}


template<typename taNumber> __global__ void StartQuiz(const StartQuizKernel<taNumber> sqk) {
  extern __shared__ DevAccumulator<taNumber> sum[];
  int64_t iInstance = threadIdx.x;
  sum[iInstance].Init(iInstance < sqk._nTargets ? sqk._pvB[iInstance] : 0);
  for(;;) {
    iInstance += blockDim.x;
    if (iInstance >= sqk._nTargets) {
      break;
    }
    sum[threadIdx.x].Add(sqk._pvB[iInstance]);
  }
  __syncthreads();
  uint32_t remains = blockDim.x >> 1;
  for (; remains > KernelLaunchContext::_cWarpSize; remains >>= 1) {
    if (threadIdx.x < remains) {
      sum[threadIdx.x].Add(sum[threadIdx.x + remains]);
    }
    __syncthreads();
  }
  for(; remains>=1; remains>>=1) {
    if (threadIdx.x < remains) {
      sum[threadIdx.x].Add(sum[threadIdx.x + remains]);
    }
  }
  __syncthreads();
  const taNumber divisor = sum[0].Get();
  iInstance = threadIdx.x;
  while (iInstance < sqk._nTargets) {
    sqk._pPriorMants[iInstance] = sqk._pvB[iInstance] / divisor;
    iInstance += blockDim.x;
  }

  const int64_t nQAskedComps = (sqk._nTargets + 31) >> 5;
  iInstance = threadIdx.x;
  while (iInstance < nQAskedComps) {
    sqk._pQAsked[iInstance] = 0;
    iInstance += blockDim.x;
  }
}

template<typename taNumber> void StartQuizKernel<taNumber>::Run(const KernelLaunchContext& klc, cudaStream_t stream) {
  const uint32_t nThreads = klc.FixBlockSize(_nTargets);
  StartQuiz<taNumber><<<1, nThreads, sizeof(DevAccumulator<taNumber>) * nThreads, stream>>>(*this);
}

//// Instantinations
template class InitStatisticsKernel<float>;
template class StartQuizKernel<float>;

} // namespace ProbQA
