// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CEUpdatePriorsSubtaskMul.h"
#include "../PqaCore/CEUpdatePriorsTask.h"
#include "../PqaCore/CEQuiz.h"
#include "../PqaCore/DoubleNumber.h"

using namespace SRPlat;

namespace ProbQA {

template<typename taNumber> CEUpdatePriorsSubtaskMul<taNumber>::CEUpdatePriorsSubtaskMul(
  CEUpdatePriorsTask<taNumber> *pTask, const TPqaId iFirstVT, const TPqaId iLimVT)
  : SRBaseSubtask(pTask), _iFirstVT(iFirstVT), _iLimVT(iLimVT)
{ }

namespace {
  const __m256i gDoubleExpMask = _mm256_set1_epi64x(0x7ffULL << 52);
  const __m256i gDoubleExp0 = _mm256_set1_epi64x(1023ULL << 52);
}

template<> template<bool taCache> void CEUpdatePriorsSubtaskMul<DoubleNumber>::RunInternal(
  const CEUpdatePriorsTask<DoubleNumber>& task)
{
  constexpr uint8_t logNumsPerVect = SRSimd::_cLogNBytes - SRMath::StaticCeilLog2(sizeof(DoubleNumber));
  auto& engine = static_cast<const CpuEngine<DoubleNumber>&>(*task.GetEngine());
  const CEQuiz<DoubleNumber> &quiz = *task._pQuiz;

  __m256d *pMants = reinterpret_cast<__m256d*>(quiz.GetTlhMants());
  static_assert(std::is_same<int64_t, CEQuiz<DoubleNumber>::TExponent>::value, "The code below assumes TExponent is"
    " 64-bit integer.");
  __m256i *pExps = reinterpret_cast<__m256i*>(quiz.GetTlhExps());

  assert(_iLimVT > _iFirstVT);
  const size_t nVectsInBlock = (taCache ? (task._nVectsInCache >> 1) : (_iLimVT - _iFirstVT));
  size_t iBlockStart = _iFirstVT;
  for(;;) {
    const size_t iBlockLim = std::min(SRCast::ToSizeT(_iLimVT), iBlockStart + nVectsInBlock);
    for (size_t i = 0; i < SRCast::ToSizeT(task._nAnswered); i++) {
      const __m256d *pAdjustments = reinterpret_cast<const __m256d*>(&engine.GetA(task._pAQs[i]._iAnswer,
        task._pAQs[i]._iQuestion, 0));
      for (size_t j = iBlockStart; j < iBlockLim; j++) {
        const __m256d oldMants = (taCache ? _mm256_load_pd(reinterpret_cast<const double*>(pMants + j))
          : _mm256_castsi256_pd(_mm256_stream_load_si256(reinterpret_cast<const __m256i*>(pMants + j))));
        const __m256d adjs = (taCache ? _mm256_load_pd(reinterpret_cast<const double*>(pAdjustments + j))
          : _mm256_castsi256_pd(_mm256_stream_load_si256(reinterpret_cast<const __m256i*>(pAdjustments + j))));
        const __m256d product = _mm256_mul_pd(oldMants, adjs);
        const __m256i prodExps = _mm256_srli_epi64(_mm256_and_si256(gDoubleExpMask, _mm256_castpd_si256(product)), 52);
        const __m256i oldExps = (taCache ? _mm256_load_si256(pExps + j) : _mm256_stream_load_si256(pExps + j));
        const __m256i newExps = _mm256_add_epi64(prodExps, oldExps);
        //TODO: verify that the branchings below are compile-time
        if (taCache) {
          _mm256_store_si256(pExps + j, newExps);
        } else {
          _mm256_stream_si256(pExps + j, newExps);
        }
        const __m256d newMants = _mm256_or_pd(_mm256_castsi256_pd(gDoubleExp0),
          _mm256_andnot_pd(_mm256_castsi256_pd(gDoubleExpMask), product));
        if (taCache) {
          _mm256_store_pd(reinterpret_cast<double*>(pMants + j), newMants);
        } else {
          _mm256_stream_pd(reinterpret_cast<double*>(pMants + j), newMants);
        }
      }
    }
    if (iBlockLim >= SRCast::ToSizeT(_iLimVT)) {
      break;
    }
    iBlockStart = iBlockLim;
  }
  if (!taCache) {
    _mm_sfence();
  }
}

template<> void CEUpdatePriorsSubtaskMul<DoubleNumber>::Run() {
  auto& task = static_cast<const CEUpdatePriorsTask<DoubleNumber>&>(*GetTask());
  // This should be a tail call
  (task._nVectsInCache < 2) ? RunInternal<false>(task) : RunInternal<true>(task);
}

template class CEUpdatePriorsSubtaskMul<DoubleNumber>;

} // namespace ProbQA
