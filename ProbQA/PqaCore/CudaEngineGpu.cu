// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include <cstdint>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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

__device__ bool TestBit(const uint8_t *pArr, const int64_t iBit) {
  return pArr[iBit >> 3] & (1 << (iBit & 7));
}

template<typename taNumber> __device__ taNumber& GetSA(const int64_t iQuestion, const int64_t iAnswer,
  const int64_t iTarget, taNumber *pSA, const int64_t nAnswers, const int64_t nTargets)
{
  return pSA[(iQuestion * nAnswers + iAnswer) * nTargets + iTarget];
}

template<typename taNumber> __device__ taNumber& GetMD(const int64_t iQuestion, const int64_t iTarget,
  taNumber *pMD, const int64_t nTargets)
{
  return pMD[iQuestion * nTargets + iTarget];
}

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
  InitStatistics<taNumber> << <nBlocks, klc.DefaultBlockSize(), /* no shared memory */ 0, stream >> > (*this);
}


template<typename taNumber> __global__ void StartQuiz(const StartQuizKernel<taNumber> sqk) {
  extern __shared__ DevAccumulator<taNumber> sum[];
  int64_t iInstance = threadIdx.x;
  sum[iInstance].Init((iInstance < sqk._nTargets && !TestBit(sqk._pTargetGaps, iInstance)) ? sqk._pvB[iInstance] : 0);
  for (;;) {
    iInstance += blockDim.x;
    if (iInstance >= sqk._nTargets) {
      break;
    }
    if (!TestBit(sqk._pTargetGaps, iInstance)) {
      sum[threadIdx.x].Add(sqk._pvB[iInstance]);
    }
  }
  __syncthreads();
  uint32_t remains = blockDim.x >> 1;
  for (; remains > KernelLaunchContext::_cWarpSize; remains >>= 1) {
    if (threadIdx.x < remains) {
      sum[threadIdx.x].Add(sum[threadIdx.x + remains]);
    }
    __syncthreads();
  }
  if (threadIdx.x < KernelLaunchContext::_cWarpSize) {
    for (; remains >= 1; remains >>= 1) {
      if (threadIdx.x < remains) {
        sum[threadIdx.x].Add(sum[threadIdx.x + remains]);
      }
    }
  }
  __syncthreads();
  const taNumber divisor = sum[0].Get();
  const taNumber multiplier = 1 / divisor;
  iInstance = threadIdx.x;
  while (iInstance < sqk._nTargets) {
    sqk._pPriorMants[iInstance] = (TestBit(sqk._pTargetGaps, iInstance) ? 0 : sqk._pvB[iInstance] * multiplier);
    iInstance += blockDim.x;
  }

  const int64_t nQAskedComps = (sqk._nTargets + 63) >> 6;
  iInstance = threadIdx.x;
  while (iInstance < nQAskedComps) {
    reinterpret_cast<uint64_t*>(sqk._pQAsked)[iInstance] = 0;
    iInstance += blockDim.x;
  }
}

template<typename taNumber> void StartQuizKernel<taNumber>::Run(const KernelLaunchContext& klc, cudaStream_t stream) {
  const uint32_t nThreads = klc.FixBlockSize(_nTargets);
  StartQuiz<taNumber> << <1, nThreads, sizeof(DevAccumulator<taNumber>) * nThreads, stream >> > (*this);
}

template<typename taNumber> struct EvaluateQuestionShared {
  DevAccumulator<taNumber> _accLhEnt;
  DevAccumulator<taNumber> _accLack;
  DevAccumulator<taNumber> _accVelocity;
};

template<typename taNumber> __device__ void EvaluateQuestion(const int64_t iQuestion,
  const NextQuestionKernel<taNumber>& nqk, EvaluateQuestionShared<taNumber> *shared)
{
  // i - questions
  // k - answers
  // j - targets
  __shared__ DevAccumulator<taNumber> accTotW;
  if (threadIdx.x == 0) {
    accTotW.Init(0);
  }
  uint32_t remains;
  for (int64_t iAnswer = 0; iAnswer < nqk._nAnswers; iAnswer++) {
    const bool isAns0 = (iAnswer == 0);
    shared[threadIdx.x]._accLhEnt.Init(0);
    shared[threadIdx.x]._accLack.Init(0);
    shared[threadIdx.x]._accVelocity.Init(0);
    for (int64_t blockFirst = 0; blockFirst < nqk._nTargets; blockFirst += blockDim.x) {
      const int64_t iTarget = threadIdx.x + blockFirst;
      if (iTarget < nqk._nTargets) {
        taNumber postLikelihood;
        if (TestBit(nqk._pTargetGaps, iTarget)) {
          postLikelihood = 0;
        }
        else {
          taNumber invCountTotal;
          if (isAns0) {
            invCountTotal = 1 / GetMD(iQuestion, iTarget, nqk._pmD, nqk._nTargets);
            nqk._pInvD[blockIdx.x*nqk._nTargets + iTarget] = invCountTotal;
          }
          else {
            invCountTotal = nqk._pInvD[blockIdx.x*nqk._nTargets + iTarget];
          }
          const taNumber Pr_Qi_eq_k_given_Tj = GetSA(iQuestion, iAnswer, iTarget, nqk._psA, nqk._nAnswers,
            nqk._nTargets) * invCountTotal;
          postLikelihood = Pr_Qi_eq_k_given_Tj * nqk._pPriorMants[iTarget];
          shared[threadIdx.x]._accLhEnt.Add(postLikelihood);
        }
        nqk._pPosteriors[blockIdx.x*nqk._nTargets + iTarget] = postLikelihood;
      }
    }
    __syncthreads(); // get the shared data in all threads
    remains = blockDim.x >> 1;
    for (; remains > KernelLaunchContext::_cWarpSize; remains >>= 1) {
      if (threadIdx.x < remains) {
        shared[threadIdx.x]._accLhEnt.Add(shared[threadIdx.x + remains]._accLhEnt);
      }
      __syncthreads();
    }
    if (threadIdx.x < KernelLaunchContext::_cWarpSize) {
      for (; remains >= 1; remains >>= 1) {
        if (threadIdx.x < remains) {
          shared[threadIdx.x]._accLhEnt.Add(shared[threadIdx.x + remains]._accLhEnt);
        }
      }
    }
    __syncthreads(); // Ensure that all threads get updated shared[0]
    const taNumber Wk = shared[0]._accLhEnt.Get();
    if (threadIdx.x == 0) {
      accTotW.Add(Wk);
      nqk._pAnsMets[blockIdx.x*nqk._nAnswers + iAnswer]._weight = Wk;
    }
    __syncthreads(); // Ensure that all threads no more need shared[0]

    const taNumber invWk = 1 / Wk;
    shared[threadIdx.x]._accLhEnt.Init(0); // reuse for entropy summation

    for (int64_t blockFirst = 0; blockFirst < nqk._nTargets; blockFirst += blockDim.x) {
      const int64_t iTarget = threadIdx.x + blockFirst;
      if (iTarget < nqk._nTargets && !TestBit(nqk._pTargetGaps, iTarget)) {
        const taNumber posterior = nqk._pPosteriors[blockIdx.x*nqk._nTargets + iTarget] * invWk;
        const taNumber prior = nqk._pPriorMants[iTarget];
        taNumber Hikj, lack, l2post;
        if (posterior == 0 || (l2post = log2(posterior)) == 0) {
          Hikj = 0;
          lack = 0;
        }
        else {
          Hikj = l2post * posterior;
          //const taNumber invDij = nqk._pInvD[blockIdx.x*nqk._nTargets + iTarget];
          //lack = invDij * invDij / l2post;
          const taNumber invSikj = 1 / GetSA(iQuestion, iAnswer, iTarget, nqk._psA, nqk._nAnswers,
            nqk._nTargets);
          lack = invSikj * invSikj / l2post;
        }
        shared[threadIdx.x]._accLhEnt.Add(Hikj);
        shared[threadIdx.x]._accLack.Add(lack);

        const taNumber diff = posterior - prior;
        const taNumber square = diff * diff;
        shared[threadIdx.x]._accVelocity.Add(square);
      }
    }
    __syncthreads(); // get the shared data in all threads
    remains = blockDim.x >> 1;
    for (; remains > KernelLaunchContext::_cWarpSize; remains >>= 1) {
      if (threadIdx.x < remains) {
        shared[threadIdx.x]._accLhEnt.Add(shared[threadIdx.x + remains]._accLhEnt);
        shared[threadIdx.x]._accLack.Add(shared[threadIdx.x + remains]._accLack);
        shared[threadIdx.x]._accVelocity.Add(shared[threadIdx.x + remains]._accVelocity);
      }
      __syncthreads();
    }
    if (threadIdx.x < KernelLaunchContext::_cWarpSize) {
      for (; remains >= 1; remains >>= 1) {
        if (threadIdx.x < remains) {
          shared[threadIdx.x]._accLhEnt.Add(shared[threadIdx.x + remains]._accLhEnt);
          shared[threadIdx.x]._accLack.Add(shared[threadIdx.x + remains]._accLack);
          shared[threadIdx.x]._accVelocity.Add(shared[threadIdx.x + remains]._accVelocity);
        }
      }
      if (threadIdx.x == 0) {
        const int64_t iAnsMet = blockIdx.x*nqk._nAnswers + iAnswer;
        nqk._pAnsMets[iAnsMet]._entropy = shared[0]._accLhEnt.Get();
        nqk._pAnsMets[iAnsMet]._lack = shared[0]._accLack.Get();
        nqk._pAnsMets[iAnsMet]._velocity = sqrt(shared[0]._accVelocity.Get());
      }
    }
  }

  shared[threadIdx.x]._accLhEnt.Init(0);
  shared[threadIdx.x]._accLack.Init(0);
  shared[threadIdx.x]._accVelocity.Init(0);
  int64_t iAnswer = threadIdx.x;
  while (iAnswer < nqk._nAnswers) {
    const int64_t iAnsMet = blockIdx.x*nqk._nAnswers + iAnswer;
    const taNumber weight = nqk._pAnsMets[iAnsMet]._weight;
    shared[threadIdx.x]._accLhEnt.Add(nqk._pAnsMets[iAnsMet]._entropy * weight);
    shared[threadIdx.x]._accLack.Add(nqk._pAnsMets[iAnsMet]._lack * weight);
    shared[threadIdx.x]._accVelocity.Add(nqk._pAnsMets[iAnsMet]._velocity * weight);
    iAnswer += blockDim.x;
  }
  __syncthreads();
  remains = blockDim.x >> 1;
  for (; remains > KernelLaunchContext::_cWarpSize; remains >>= 1) {
    if (threadIdx.x < remains) {
      shared[threadIdx.x]._accLhEnt.Add(shared[threadIdx.x + remains]._accLhEnt);
      shared[threadIdx.x]._accLack.Add(shared[threadIdx.x + remains]._accLack);
      shared[threadIdx.x]._accVelocity.Add(shared[threadIdx.x + remains]._accVelocity);
    }
    __syncthreads();
  }
  if (threadIdx.x < KernelLaunchContext::_cWarpSize) {
    for (; remains >= 1; remains >>= 1) {
      if (threadIdx.x < remains) {
        shared[threadIdx.x]._accLhEnt.Add(shared[threadIdx.x + remains]._accLhEnt);
        shared[threadIdx.x]._accLack.Add(shared[threadIdx.x + remains]._accLack);
        shared[threadIdx.x]._accVelocity.Add(shared[threadIdx.x + remains]._accVelocity);
      }
    }
    if (threadIdx.x == 0) {
      const taNumber totW = accTotW.Get(); // actually this must be equal to 1 (+-)
      const taNumber normalizer = 1 / totW;
      const taNumber avgH = -shared[0]._accLhEnt.Get() * normalizer;
      const taNumber avgL = -shared[0]._accLack.Get() * normalizer;
      const taNumber avgV = shared[0]._accVelocity.Get() * normalizer;
      const taNumber nExpectedTargets = exp2(avgH);
      nqk._pTotals[iQuestion] = pow(avgL, 1) * pow(avgV + 1e-4, 9) * pow(nExpectedTargets, -2);
    }
  }
}

template<typename taNumber> __global__ void NextQuestion(const NextQuestionKernel<taNumber> nqk) {
  extern __shared__ EvaluateQuestionShared<taNumber> shared[];
//  __shared__ int64_t prevTime;
  int64_t iQuestion = blockIdx.x;
  while (iQuestion < nqk._nQuestions) {
    if ( TestBit(nqk._pQAsked, iQuestion) || TestBit(nqk._pQuestionGaps, iQuestion) ) {
      if (threadIdx.x == 0) {
        nqk._pTotals[iQuestion] = 0;
      }
    }
    else {
      EvaluateQuestion(iQuestion, nqk, shared);
    }
    iQuestion += gridDim.x;
  }
}

template<typename taNumber> void NextQuestionKernel<taNumber>::Run(cudaStream_t stream)
{
  NextQuestion<taNumber><<<_nBlocks, _nThreadsPerBlock, sizeof(EvaluateQuestionShared<taNumber>) * _nThreadsPerBlock,
    stream>>>(*this);
}


template<typename taNumber> __device__ taNumber GetUpdatedPrior(const RecordAnswerKernel<taNumber>& rak,
  const int64_t iTarget)
{
  return rak._pPriorMants[iTarget] * GetSA(rak._iQuestion, rak._iAnswer, iTarget, rak._psA, rak._nAnswers,
    rak._nTargets) / GetMD(rak._iQuestion, iTarget, rak._pmD, rak._nTargets);
}

template<typename taNumber> __global__ void RecordAnswer(RecordAnswerKernel<taNumber> rak) {
  extern __shared__ DevAccumulator<taNumber> sum[];
  int64_t iTarget = threadIdx.x;
  if (iTarget < rak._nTargets) {
    taNumber updPrior = GetUpdatedPrior(rak, iTarget);
    sum[iTarget].Init(updPrior);
    rak._pPriorMants[iTarget] = updPrior;
    for (;;) {
      iTarget += blockDim.x;
      if (iTarget >= rak._nTargets) {
        break;
      }
      updPrior = GetUpdatedPrior(rak, iTarget);
      sum[iTarget].Add(updPrior);
      rak._pPriorMants[iTarget] = updPrior;
    }
  }
  else {
    sum[iTarget].Init(0);
  }
  __syncthreads();
  uint32_t remains = blockDim.x >> 1;
  for (; remains > KernelLaunchContext::_cWarpSize; remains >>= 1) {
    if (threadIdx.x < remains) {
      sum[threadIdx.x].Add(sum[threadIdx.x + remains]);
    }
    __syncthreads();
  }
  if (threadIdx.x < KernelLaunchContext::_cWarpSize) {
    for (; remains >= 1; remains >>= 1) {
      if (threadIdx.x < remains) {
        sum[threadIdx.x].Add(sum[threadIdx.x + remains]);
      }
    }
  }
  __syncthreads();
  const taNumber normalizer = 1 / sum[0].Get();
  iTarget = threadIdx.x;
  while (iTarget < rak._nTargets) {
    rak._pPriorMants[iTarget] *= normalizer;
    iTarget += blockDim.x;
  }
}

template<typename taNumber> void RecordAnswerKernel<taNumber>::Run(const KernelLaunchContext& klc, cudaStream_t stream)
{
  const uint32_t nThreads = klc.FixBlockSize(_nTargets);
  RecordAnswer<taNumber><<<1, nThreads, sizeof(DevAccumulator<taNumber>)*nThreads, stream>>>(*this);
}


template<typename taNumber> __global__ void RecordQuizTarget(const RecordQuizTargetKernel<taNumber> rqtk) {
  const uint32_t nThreads = blockDim.x * gridDim.x;
  int64_t iAQ = threadIdx.x + blockIdx.x*blockDim.x;
  if (iAQ == 0) {
    rqtk._pvB[rqtk._iTarget] += rqtk._amount;
  }
  while (iAQ < rqtk._nAQs) {
    const int64_t iQuestion = rqtk._pAQs[iAQ]._iQuestion;
    if (iAQ == 0 || iQuestion != rqtk._pAQs[iAQ - 1]._iQuestion) {
      int64_t iSame = iAQ;
      do {
        const int64_t iAnswer = rqtk._pAQs[iSame]._iAnswer;
        taNumber& Aikj = GetSA(iQuestion, iAnswer, rqtk._iTarget, rqtk._psA, rqtk._nAnswers, rqtk._nTargets);;
        const taNumber aSquare = Aikj;
        const taNumber a = sqrt(aSquare);
        const taNumber addend = a * rqtk._twoB + rqtk._bSquare;
        Aikj += addend;
        GetMD(iQuestion, rqtk._iTarget, rqtk._pmD, rqtk._nTargets) += addend;
        iSame++;
      } while (iSame < rqtk._nAQs && rqtk._pAQs[iSame]._iQuestion == iQuestion);
    }
    iAQ += nThreads;
  }
}

template<typename taNumber> void RecordQuizTargetKernel<taNumber>::Run(const KernelLaunchContext& klc,
  cudaStream_t stream)
{
  const uint32_t nThreads = klc.FixBlockSize(_nTargets);
  const uint32_t nBlocks = klc.GetBlockCount(_nAQs, nThreads);
  RecordQuizTarget<taNumber><<<nBlocks, nThreads, /* no shared memory */ 0, stream>>>(*this);
}

//// Instantinations
template struct InitStatisticsKernel<float>;
template struct StartQuizKernel<float>;
template struct NextQuestionKernel<float>;
template struct RecordAnswerKernel<float>;
template struct RecordQuizTargetKernel<float>;

} // namespace ProbQA
