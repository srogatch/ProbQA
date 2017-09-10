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

template<> void CENormPriorsSubtaskMax<SRDoubleNumber>::Run() {
  auto const& task = static_cast<const CENormPriorsTask<SRDoubleNumber>&>(*GetTask());
  auto const& engine = static_cast<const CpuEngine<SRDoubleNumber>&>(task.GetBaseEngine());
  const __m256i *pExps = SRCast::CPtr<__m256i>(task._pQuiz->GetTlhExps());
  const __m256d *pMants = SRCast::CPtr<__m256d>(task._pQuiz->GetTlhMants());
  const bool isAtPartial = (_iLimit + 1 == task._iPartial);

  __m256i curMax = _mm256_set1_epi64x(std::numeric_limits<int64_t>::min());
  for (TPqaId i = _iFirst, iEn = (isAtPartial ? task._iPartial : _iLimit); i < iEn; i++) {
    const __m256i totExp = _mm256_add_epi64(SRSimd::Load<false, __m256i>(pExps + i),
      SRSimd::ExtractExponents64<false>(SRSimd::Load<false, __m256d>(pMants+i)));

    // Mask away the targets at gaps
    const uint8_t gaps = engine.GetTargetGaps().GetQuad(i);
    const __m256i cmMask = SRSimd::SetToBitQuad(gaps);

    curMax = SRSimd::MaxI64(curMax, totExp, cmMask);
  }
  if (isAtPartial) {

  }
}


template class CENormPriorsSubtaskMax<SRPlat::SRDoubleNumber>;

} // namespace ProbQA
