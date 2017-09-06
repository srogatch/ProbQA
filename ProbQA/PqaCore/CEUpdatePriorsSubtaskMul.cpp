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

template<> template<bool taCache> void CEUpdatePriorsSubtaskMul<DoubleNumber>::RunInternal(
  const CEUpdatePriorsTask<DoubleNumber>& task) const
{
  constexpr uint8_t logNumsPerVect = SRSimd::_cLogNBytes - SRMath::StaticCeilLog2(sizeof(DoubleNumber));
  auto& engine = static_cast<const CpuEngine<DoubleNumber>&>(task.GetBaseEngine());
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
      const AnsweredQuestion& aq = task._pAQs[i];
      const __m256d *pAdjMuls = reinterpret_cast<const __m256d*>(&engine.GetA(aq._iQuestion, aq._iAnswer, 0));
      const __m256d *pAdjDivs = reinterpret_cast<const __m256d*>(&engine.GetD(aq._iQuestion, 0));
      for (size_t j = iBlockStart; j < iBlockLim; j++) {
        const __m256d adjMuls = SRSimd::Load<false>(pAdjMuls + j);
        const __m256d adjDivs = SRSimd::Load<false>(pAdjDivs + j);
        // P(answer(aq._iQuestion)==aq._iAnswer GIVEN target==(j0,j1,j2,j3))
        const __m256d P_qa_given_t = _mm256_div_pd(adjMuls, adjDivs);

        //TODO: verify that taCache based branchings are compile-time
        const __m256d oldMants = SRSimd::Load<taCache>(pMants + j);
        const __m256d product = _mm256_mul_pd(oldMants, P_qa_given_t);
        //TODO: move separate summation of exponent to a common function (available to other subtasks etc.)?
        const __m256d newMants = _mm256_or_pd(_mm256_castsi256_pd(SRSimd::_cDoubleExp0),
          _mm256_andnot_pd(_mm256_castsi256_pd(SRSimd::_cDoubleExpMask), product));
        SRSimd::Store<taCache>(pMants + j, newMants);

        //TODO: AND can be removed here if numbers are non-negative or we can assume a large exponent for negatives
        const __m256i prodExps = _mm256_srli_epi64(
          _mm256_and_si256(SRSimd::_cDoubleExpMask, _mm256_castpd_si256(product)), 52);
        const __m256i oldExps = SRSimd::Load<taCache>(pExps+j);
        const __m256i newExps = _mm256_add_epi64(prodExps, oldExps);
        SRSimd::Store<taCache>(pExps + j, newExps);
      }
    }
    if (taCache) {
      const size_t nBytes = (iBlockLim - iBlockStart) << SRSimd::_cLogNBytes;
      // The bounds may be used in the next block or by another thread.
      if (iBlockStart > SRCast::ToSizeT(_iFirstVT)) {
        // Can flush left because it's for the current thread only and has been processed.
        SRUtils::FlushCache<true, false>(pMants + iBlockStart, nBytes);
        SRUtils::FlushCache<true, false>(pExps + iBlockStart, nBytes);
      } else {
        // Can't flush left because another thread may be using it
        SRUtils::FlushCache<false, false>(pMants + iBlockStart, nBytes);
        SRUtils::FlushCache<false, false>(pExps + iBlockStart, nBytes);
      }
      _mm_sfence();
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
