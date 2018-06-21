// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

namespace ProbQA {

struct KernelLaunchContext {
  static const uint8_t _cLogBlockSize = 8;
  static const uint8_t _cLogWarpSize = 5;
  static const uint32_t _cWarpSize = (1 << _cLogWarpSize);
  static const uint64_t _cWarpMask = _cWarpSize - 1;

  uint32_t _maxBlocks;
  cudaDeviceProp _cdp;

  uint32_t GetBlockCount(const int64_t nInstances) const {
    return uint32_t(std::min(((nInstances - 1) >> _cLogBlockSize) + 1, int64_t(_maxBlocks)));
  }

  uint32_t DefaultBlockSize() const { return 1ui32 << _cLogBlockSize; }

  uint32_t FixBlockSize(const int64_t rawSize) const {
    const uint32_t bound = uint32_t(std::min(rawSize, int64_t(_cdp.maxThreadsPerBlock)));
    unsigned long msb;
    if (_BitScanReverse(&msb, bound)) {
      const uint32_t t = (1 << msb);
      if (bound == t) {
        return t;
      }
      else {
        return t << 1;
      }
    }
    return 0;
  }
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

template<typename taNumber> struct CudaAnswerMetrics {
  taNumber _weight;
  taNumber _entropy;
  taNumber _lack;
  taNumber _velocity;
};

template<typename taNumber> struct NextQuestionKernel {
  //// Inputs
  int64_t _nQuestions;
  int64_t _nAnswers;
  int64_t _nTargets;
  taNumber *_psA;
  taNumber *_pmD;
  uint32_t *_pQAsked;
  taNumber *_pPriorMants;
  uint32_t *_pTargetGaps;
  uint32_t *_pQuestionGaps;
  uint32_t _nThreadsPerBlock;
  uint32_t _nBlocks;
  
  //// Work arrays
  taNumber *_pPosteriors; // nBlocks * nTargets
  taNumber *_pInvD; // nBlocks * nTargets
  CudaAnswerMetrics<taNumber> *_pAnsMets; // nBlocks * nAnswers

  //// Outputs
  taNumber *_pTotals;

  void Run(cudaStream_t stream);
};

} // namespace ProbQA
