// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CENormPriorsSubtaskMax.h"
#include "../PqaCore/CENormPriorsTask.h"
#include "../PqaCore/CEQuiz.h"

using namespace SRPlat;

namespace ProbQA {

template class CENormPriorsSubtaskMax<SRDoubleNumber>;

template<typename taNumber> CENormPriorsSubtaskMax<taNumber>::CENormPriorsSubtaskMax(TTask *pTask)
  : SRStandardSubtask(pTask) { }

namespace {

struct ContextDouble {
  const __m256d *PTR_RESTRICT _pMants;
  const __m256i *PTR_RESTRICT _pExps;
  const GapTracker<TPqaId> *PTR_RESTRICT _pGt;

  // Returns the total exponents.
  // |retention| is an output parameter for the mask for retaining the old maximum.
  ATTR_NOALIAS inline __m256i __vectorcall Process(const TPqaId iVect, __m256i &PTR_RESTRICT retention) {
    const __m256i totExp = _mm256_add_epi64(SRSimd::Load<false, __m256i>(_pExps + iVect),
      SRSimd::ExtractExponents64<false>(SRSimd::Load<false, __m256d>(_pMants + iVect)));
    // Mask away the targets at gaps
    const uint8_t gaps = _pGt->GetQuad(iVect);
    retention = SRSimd::SetToBitQuadHot(gaps);
    return totExp;
  }
};

} // anonymous namespace

template<> void CENormPriorsSubtaskMax<SRDoubleNumber>::Run() {
  auto &PTR_RESTRICT task = static_cast<const TTask&>(*GetTask());
  auto &PTR_RESTRICT engine = static_cast<const CpuEngine<SRDoubleNumber>&>(task.GetBaseEngine());
  const CEQuiz<SRDoubleNumber> &PTR_RESTRICT quiz = task.GetQuiz();

  ContextDouble ctx;
  ctx._pGt = &engine.GetTargetGaps();
  ctx._pExps = SRCast::CPtr<__m256i>(quiz.GetTlhExps());
  ctx._pMants = SRCast::CPtr<__m256d>(quiz.GetTlhMants());

  __m256i curMax = _mm256_set1_epi64x(std::numeric_limits<int64_t>::min());
  for (TPqaId i = _iFirst, iEn = _iLimit; i < iEn; i++) {
    __m256i retention;
    const __m256i totExp = ctx.Process(i, retention);
    curMax = SRSimd::MaxI64(curMax, totExp, retention);
  }
  _maxExp = SRSimd::FullHorizMaxI64(curMax);
}

} // namespace ProbQA
