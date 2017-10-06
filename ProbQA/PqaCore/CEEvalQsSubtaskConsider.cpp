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

#define LOCLOG(severityVar) SRLogStream(ISRLogger::Severity::severityVar, engine.GetLogger())

template<> void CEEvalQsSubtaskConsider<SRDoubleNumber>::Run() {
  auto &PTR_RESTRICT task = static_cast<const TTask&>(*GetTask());
  auto &PTR_RESTRICT engine = static_cast<const CpuEngine<SRDoubleNumber>&>(task.GetBaseEngine());
  const CEQuiz<SRDoubleNumber> &PTR_RESTRICT quiz = task.GetQuiz();
  auto *PTR_RESTRICT pPriors = SRCast::CPtr<__m256d>(quiz.GetPriorMants());
  SRBucketSummatorSeq<SRDoubleNumber> bss(
    task._pBSes + SRBucketSummatorSeq<SRDoubleNumber>::GetMemoryRequirementBytes() * _iWorker);
  __m256d *PTR_RESTRICT pPosteriors = SRCast::Ptr<__m256d>(task._pPosteriors + task._threadPosteriorBytes * _iWorker);
  double *PTR_RESTRICT pAnswerWeigths = SRCast::Ptr<double>(task._pAnswerMetrics
    + task._threadAnswerMetricsBytes * _iWorker);
  double *PTR_RESTRICT pAnswerEntropies = SRCast::Ptr<double>(task._pAnswerMetrics
    + task._threadAnswerMetricsBytes * _iWorker + (task._threadAnswerMetricsBytes>>1));

  const TPqaId nTargVects = SRMath::RShiftRoundUp(engine.GetDims()._nTargets, SRSimd::_cLogNComps64);
  double prevRunLength = 0;
  for (TPqaId i = _iFirst; i < _iLimit; i++) {
    if (engine.GetQuestionGaps().IsGap(i) || SRBitHelper::Test(quiz.GetQAsked(), i)) {
      // Set 0 probability to this question
      task._pRunLength[i].SetValue(prevRunLength);
      continue;
    }
    SRAccumulator<SRDoubleNumber> accTotW(SRDoubleNumber(0.0));
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
      accTotW.Add(SRDoubleNumber(Wk));
      pAnswerWeigths[k] = Wk;
      const __m256d vWk = _mm256_set1_pd(Wk);
      for (TPqaId j = 0; j < nTargVects; j++) {
        const __m256d targProb = _mm256_div_pd(SRSimd::Load<false>(pPosteriors + j), vWk);
        // Calculate negated entropy component: negated self-information multiplied by probability of its event.
        const __m256d Hikj = _mm256_mul_pd(targProb, SRVectMath::Log2Hot(targProb));
        const __m256i gapMask = SRSimd::SetToBitQuadHot(engine.GetTargetGaps().GetQuad(j));
        const __m256d maskedHikj = _mm256_andnot_pd(_mm256_castsi256_pd(gapMask), Hikj);
        bss.CalcAdd(maskedHikj);
      }
      const double entropyHik = -bss.ComputeSum().GetValue();
      pAnswerEntropies[k] = entropyHik;
    }
    const double totW = accTotW.Get().GetValue();
    if (std::fabs(totW - 1.0) > 1e-9) {
      LOCLOG(Warning) << SR_FILE_LINE "The sum of answer weights is " << totW;
    }
    SRAccumulator<SRDoubleNumber> accAvgH(SRDoubleNumber(0.0));
    const TPqaId nAnswerVects = engine.GetDims()._nAnswers >> SRSimd::_cLogNComps64;
    for (TPqaId vk = 0; vk < nAnswerVects; vk++) {
      const __m256d curW = SRSimd::Load<true>(SRCast::CPtr<__m256d>(pAnswerWeigths) + vk);
      const __m256d curH = SRSimd::Load<true>(SRCast::CPtr<__m256d>(pAnswerEntropies) + vk);
      const __m256d product = _mm256_mul_pd(curW, curH);
      accAvgH.Add(SRDoubleNumber(product.m256d_f64[0])).Add(SRDoubleNumber(product.m256d_f64[1]))
        .Add(SRDoubleNumber(product.m256d_f64[2])).Add(SRDoubleNumber(product.m256d_f64[3]));
    }
    for (TPqaId k = (nAnswerVects << SRSimd::_cLogNComps64); k < engine.GetDims()._nAnswers; k++) {
      accAvgH.Add(SRDoubleNumber(pAnswerWeigths[k] * pAnswerEntropies[k]));
    }

    // The average entropy over all answers for this question
    const double avgH = accAvgH.Get().GetValue() / totW;
    const double nExpectedTargets = std::exp2(avgH);
    const double cutoff = task._nValidTargets - nExpectedTargets;
    double priority;
    if (cutoff  < -1e-9) {
      LOCLOG(Warning) << SR_FILE_LINE "The expected number of targets (according to entropy) is " << nExpectedTargets
        << ", while the actual number of valid targets is " << task._nValidTargets;
      priority = 1.0;
    }
    else {
      priority = 1 + cutoff;
    }
    prevRunLength += priority;
    task._pRunLength[i].SetValue(prevRunLength);
  }
}

} // namespace ProbQA
