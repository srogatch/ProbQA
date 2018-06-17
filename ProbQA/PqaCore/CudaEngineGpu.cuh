// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

namespace ProbQA {

struct KernelLaunchContext {
  int32_t _maxBlocks;
  uint8_t _logBlockSize;

  int32_t GetBlockCount(const int64_t nInstances) const {
    return int32_t(std::min(((nInstances - 1) >> _logBlockSize) + 1, int64_t(_maxBlocks)));
  }
  int32_t GetBlockSize() const { return 1 << _logBlockSize; }
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

} // namespace ProbQA
