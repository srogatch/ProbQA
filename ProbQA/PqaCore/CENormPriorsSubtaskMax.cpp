// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CENormPriorsSubtaskMax.h"
#include "../PqaCore/CENormPriorsTask.h"
#include "../PqaCore/CEQuiz.h"

using namespace SRPlat;

namespace ProbQA {

template<typename taNumber> CENormPriorsSubtaskMax<taNumber>::CENormPriorsSubtaskMax(CENormPriorsTask<taNumber> *pTask)
  : SRStandardSubtask(pTask)
{ }

namespace {

// Returns the total exponents.
// |retention| is an output parameter for the mask for retaining the old maximum.
ATTR_NOALIAS inline __m256i __vectorcall Process(const __m256d *PTR_RESTRICT pMants, const __m256i *PTR_RESTRICT pExps,
  __m256i &PTR_RESTRICT retention, const uint8_t gaps)
{
  const __m256i totExp = _mm256_add_epi64(SRSimd::Load<false, __m256i>(pExps),
    SRSimd::ExtractExponents64<false>(SRSimd::Load<false, __m256d>(pMants)));
  // Mask away the targets at gaps
  retention = SRSimd::SetToBitQuadHot(gaps);
  return totExp;
}

} // anonymous namespace

template<> void CENormPriorsSubtaskMax<SRDoubleNumber>::Run() {
  auto const& task = static_cast<const CENormPriorsTask<SRDoubleNumber>&>(*GetTask());
  auto const& engine = static_cast<const CpuEngine<SRDoubleNumber>&>(task.GetBaseEngine());
  const GapTracker<TPqaId>& gapTracker = engine.GetTargetGaps();
  const __m256i *pExps = SRCast::CPtr<__m256i>(task._pQuiz->GetTlhExps());
  const __m256d *pMants = SRCast::CPtr<__m256d>(task._pQuiz->GetTlhMants());
  const bool isAtPartial = (_iLimit + 1 == task._iPartial);

  __m256i curMax = _mm256_set1_epi64x(std::numeric_limits<int64_t>::min());
  for (TPqaId i = _iFirst, iEn = (isAtPartial ? task._iPartial : _iLimit); i < iEn; i++) {
    __m256i retention;
    const __m256i totExp = Process(pMants + i, pExps + i, retention, gapTracker.GetQuad(i));
    curMax = SRSimd::MaxI64(curMax, totExp, retention);
  }
  if (isAtPartial) {
    __m256i retention;
    const __m256i totExp = Process(pMants + task._iPartial, pExps + task._iPartial, retention,
      gapTracker.GetQuad(task._iPartial));
    retention = _mm256_or_si256(retention, SRSimd::SetHighComps64(SRNumPack<SRDoubleNumber>::_cnComps - task._nValid));
    curMax = SRSimd::MaxI64(curMax, totExp, retention);
  }
  _maxExp = SRSimd::FullHorizMaxI64(curMax);
}


template class CENormPriorsSubtaskMax<SRPlat::SRDoubleNumber>;

} // namespace ProbQA
