// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

namespace ProbQA {

struct KernelLaunchContext {
  uint32_t _maxBlocks;
  uint8_t _logBlockSize;
  cudaDeviceProp _cdp;

  uint32_t GetBlockCount(const int64_t nInstances) const {
    return uint32_t(std::min(((nInstances - 1) >> _logBlockSize) + 1, int64_t(_maxBlocks)));
  }
  uint32_t GetBlockSize() const { return 1ui32 << _logBlockSize; }
};

template<typename taNumber> struct InitStatisticsKernel {
  taNumber *_psA;
  taNumber *_pmD;
  taNumber *_pvB;
  int64_t _nSAItems;
  int64_t _nMDItems;
  int64_t _nVBItems;
  taNumber _init1;
  taNumber _initSqr;
  taNumber _initMD;

  void Run(const KernelLaunchContext& klc, cudaStream_t stream);
};

template<typename taNumber> struct StartQuizKernel {
  uint32_t *_pQAsked;
  taNumber *_pPriorMants;
  int64_t _nTargets;
  taNumber *_pvB;
  void Run(const KernelLaunchContext& klc, cudaStream_t stream);
};

} // namespace ProbQA
