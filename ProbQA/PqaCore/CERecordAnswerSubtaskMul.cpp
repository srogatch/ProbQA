// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CERecordAnswerSubtaskMul.h"
#include "../PqaCore/CEQuiz.h"

using namespace SRPlat;

namespace ProbQA {

template class CERecordAnswerSubtaskMul<SRDoubleNumber>;

template<> void CERecordAnswerSubtaskMul<SRDoubleNumber>::Run() {
  auto &PTR_RESTRICT  task = static_cast<const TTask&>(*GetTask());
  auto &PTR_RESTRICT engine = static_cast<const CpuEngine<SRDoubleNumber>&>(task.GetBaseEngine());
  const CEQuiz<SRDoubleNumber> &PTR_RESTRICT quiz = task.GetQuiz();
  SRBucketSummatorPar<SRDoubleNumber> &PTR_RESTRICT bsp = task.GetBSP();
  const GapTracker<TPqaId>& targGaps = engine.GetTargetGaps();

  __m256d *PTR_RESTRICT pMants = SRCast::Ptr<__m256d>(quiz.GetPriorMants());
  static_assert(std::is_same<int64_t, CEQuiz<SRDoubleNumber>::TExponent>::value, "The code below assumes TExponent is"
    " 64-bit integer.");

  const AnsweredQuestion &PTR_RESTRICT aq = task.GetAQ();
  const __m256d *PTR_RESTRICT pAdjMuls = SRCast::CPtr<__m256d>(&engine.GetA(aq._iQuestion, aq._iAnswer, 0));
  const __m256d *PTR_RESTRICT pAdjDivs = SRCast::CPtr<__m256d>(&engine.GetD(aq._iQuestion, 0));
  for (TPqaId i = _iFirst; i < _iLimit; i++) {
    const __m256d adjMuls = SRSimd::Load<false>(pAdjMuls + i);
    const __m256d adjDivs = SRSimd::Load<false>(pAdjDivs + i);
    // P(answer(aq._iQuestion)==aq._iAnswer GIVEN target==(j0,j1,j2,j3))
    const __m256d P_qa_given_t = _mm256_div_pd(adjMuls, adjDivs);

    const __m256d oldMants = SRSimd::Load<false>(pMants + i);
    const __m256d product = _mm256_mul_pd(oldMants, P_qa_given_t);
    const uint8_t gaps = targGaps.GetQuad(i);
    const __m256d newMants = _mm256_andnot_pd(_mm256_castsi256_pd(SRSimd::SetToBitQuadHot(gaps)), product);
    SRSimd::Store<false>(pMants + i, newMants);

    bsp.CalcAdd(_iWorker, newMants);
  }
}

} // namespace ProbQA
