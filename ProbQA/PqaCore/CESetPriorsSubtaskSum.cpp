// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CESetPriorsSubtaskSum.h"
#include "../PqaCore/CESetPriorsTask.h"
#include "../PqaCore/CpuEngine.h"
#include "../PqaCore/CEQuiz.h"

using namespace SRPlat;

namespace ProbQA {

template class CESetPriorsSubtaskSum<SRDoubleNumber>;

template<> void CESetPriorsSubtaskSum<SRDoubleNumber>::Run() {
  auto &PTR_RESTRICT task = static_cast<const TTask&>(*GetTask());
  auto &PTR_RESTRICT engine = static_cast<const CpuEngine<SRDoubleNumber>&>(task.GetBaseEngine());
  const CEQuiz<SRDoubleNumber> &PTR_RESTRICT quiz = task.GetQuiz();
  SRBucketSummatorPar<SRDoubleNumber> &PTR_RESTRICT bsp = task.GetBSP();
  const GapTracker<TPqaId> &PTR_RESTRICT targGaps = engine.GetTargetGaps();

  static_assert(std::is_same<int64_t, CEQuiz<SRDoubleNumber>::TExponent>::value, "The code below assumes TExponent is"
    " 64-bit integer.");
  auto *PTR_RESTRICT pExps = SRCast::Ptr<__m256i>(quiz.GetTlhExps());
  auto *PTR_RESTRICT pMants = SRCast::Ptr<__m256d>(quiz.GetPriorMants());
  auto *PTR_RESTRICT pvB = SRCast::CPtr<__m256d>(&(engine.GetB(0)));

  bsp.ZeroBuckets(_iWorker);
  for (TPqaId i = _iFirst; i < _iLimit; i++) {
    const __m256d allMants = SRSimd::Load<false>(pvB + i);
    const uint8_t gaps = targGaps.GetQuad(i);
    const __m256d activeMants = _mm256_andnot_pd(_mm256_castsi256_pd(SRSimd::SetToBitQuadHot(gaps)), allMants);
    SRSimd::Store<false>(pMants + i, activeMants);
    SRSimd::Store<false>(pExps + i, _mm256_setzero_si256());
    bsp.CalcAdd(_iWorker, activeMants);
  }
  _mm_sfence();
}

} // namespace ProbQA
