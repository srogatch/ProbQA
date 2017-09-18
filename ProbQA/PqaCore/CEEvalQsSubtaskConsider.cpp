// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CEEvalQsSubtaskConsider.h"
#include "../PqaCore/CEEvalQsTask.h"
#include "../PqaCore/CEQuiz.h"

using namespace SRPlat;

namespace ProbQA {

template class CEEvalQsSubtaskConsider<SRDoubleNumber>;

template<typename taNumber> CEEvalQsSubtaskConsider<taNumber>::CEEvalQsSubtaskConsider(TTask *pTask)
  : SRStandardSubtask(pTask) { }

template<> void CEEvalQsSubtaskConsider<SRDoubleNumber>::Run() {
  auto &PTR_RESTRICT task = static_cast<const TTask&>(*GetTask());
  auto &PTR_RESTRICT engine = static_cast<const CpuEngine<SRDoubleNumber>&>(task.GetBaseEngine());
  const CEQuiz<SRDoubleNumber> &PTR_RESTRICT quiz = task.GetQuiz();
  const __m256d *pPriors = SRCast::CPtr<__m256d>(quiz.GetTlhMants());
  SRBucketSummatorSeq<SRDoubleNumber> bss(
    task._pBSes + SRBucketSummatorSeq<SRDoubleNumber>::GetMemoryRequirementBytes() * _iWorker);
  __m256d *pPosteriors = SRCast::Ptr<__m256d>(task._pPosteriors + task._threadPosteriorBytes * _iWorker);

  const TPqaId nTargVects = SRMath::RShiftRoundUp(engine.GetDims()._nTargets, SRSimd::_cLogNComps64);
  double prevRunLength = 0;
  for (TPqaId i = _iFirst; i < _iLimit; i++) {
    if (engine.GetQuestionGaps().IsGap(i) || SRBitHelper::Test(quiz.GetQAsked(), i)) {
      // Set 0 probability to this question
      task._pRunLength[i].SetValue(prevRunLength);
      continue;
    }
    for (TPqaId k = 0; k < engine.GetDims()._nAnswers; k++) {
      bss.ZeroBuckets();
      const __m256d *psAik = SRCast::CPtr<__m256d>(&engine.GetA(i, k, 0));
      const __m256d *pmDi = SRCast::CPtr<__m256d>(&engine.GetD(i, 0));
      for (TPqaId j = 0; j < nTargVects; j++) {
        const __m256d Pr_Qi_eq_k_given_Tj = _mm256_div_pd(SRSimd::Load<false>(psAik+j), SRSimd::Load<false>(pmDi+j));
        const __m256d likelihood = _mm256_mul_pd(Pr_Qi_eq_k_given_Tj, SRSimd::Load<false>(pPriors + j));
        const __m256i gapMask = SRSimd::SetToBitQuadHot(engine.GetTargetGaps().GetQuad(j));
        const __m256d maskedLH = _mm256_andnot_pd(_mm256_castsi256_pd(gapMask), likelihood);
        SRSimd::Store<false>(pPosteriors + j, maskedLH);
        bss.CalcAdd(maskedLH);
      }
      const double Wk = bss.ComputeSum().GetValue();

      bss.ZeroBuckets();
      const __m256d vWk = _mm256_set1_pd(Wk);
      for (TPqaId j = 0; j < nTargVects; j++) {
        const __m256d targProb = _mm256_div_pd(SRSimd::Load<false>(pPosteriors + j), vWk);
        //TODO: calculate self-information content here after implementing SIMD log2 function.
      }
    }
  }
  //TODO: implement
}

} // namespace ProbQA
