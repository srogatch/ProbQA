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
  __m256d *PTR_RESTRICT pPosteriors = SRCast::Ptr<__m256d>(task._pPosteriors + task._threadPosteriorBytes * _iWorker);
  const TPqaId nAnswers = engine.GetDims()._nAnswers;
  AnswerMetrics<SRDoubleNumber> *PTR_RESTRICT pAnsMet = task._pAnswerMetrics + _iWorker * nAnswers;

  const TPqaId nTargVects = SRMath::RShiftRoundUp(engine.GetDims()._nTargets, SRSimd::_cLogNComps64);
  SRAccumulator<SRDoubleNumber> accRunLength(SRDoubleNumber(0.0));
  for (TPqaId i = _iFirst; i < _iLimit; i++) {
    if (engine.GetQuestionGaps().IsGap(i) || SRBitHelper::Test(quiz.GetQAsked(), i)) {
      // Set 0 probability to this question
      task._pRunLength[i] = accRunLength.Get();
      continue;
    }
    SRAccumulator<SRDoubleNumber> accTotW(SRDoubleNumber(0.0));
    for (TPqaId k = 0; k < nAnswers; k++) {
      SRAccumVectDbl256 accVect;
      const __m256d *psAik = SRCast::CPtr<__m256d>(&engine.GetA(i, k, 0));
      const __m256d *pmDi = SRCast::CPtr<__m256d>(&engine.GetD(i, 0));
      for (TPqaId j = 0; j < nTargVects; j++) {
        const __m256d Pr_Qi_eq_k_given_Tj = _mm256_div_pd(SRSimd::Load<false>(psAik+j), SRSimd::Load<false>(pmDi+j));
        const __m256d likelihood = _mm256_mul_pd(Pr_Qi_eq_k_given_Tj, SRSimd::Load<false>(pPriors + j));
        const uint8_t gaps = engine.GetTargetGaps().GetQuad(j);
        const __m256i gapMask = SRSimd::SetToBitQuadHot(gaps);
        const __m256d maskedLH = _mm256_andnot_pd(_mm256_castsi256_pd(gapMask), likelihood);
        SRSimd::Store<false>(pPosteriors + j, maskedLH);
        //bss.CalcAdd(maskedLH);
        accVect.Add(maskedLH);
      }
      const double Wk = accVect.GetFullSum();
      accTotW.Add(SRDoubleNumber(Wk));
      pAnsMet[k]._weight.SetValue(Wk);
      const __m256d vWk = _mm256_set1_pd(Wk);

      accVect.Reset(); // reuse for entropy summation
      SRAccumVectDbl256 accDist;
      for (TPqaId j = 0; j < nTargVects; j++) {
        const __m256d posteriors = _mm256_div_pd(SRSimd::Load<false>(pPosteriors + j), vWk);
        const __m256d priors = SRSimd::Load<false>(pPriors + j);

        const uint8_t gaps = engine.GetTargetGaps().GetQuad(j);
        const __m256i gapMask = SRSimd::SetToBitQuadHot(gaps);

        // Calculate negated entropy component: negated self-information multiplied by probability of its event.
        const __m256d Hikj = _mm256_mul_pd(posteriors, SRVectMath::Log2Hot(posteriors));
        const __m256d maskedHikj = _mm256_andnot_pd(_mm256_castsi256_pd(gapMask), Hikj);
        accVect.Add(maskedHikj);

        const __m256d diff = _mm256_sub_pd(posteriors, priors);
        const __m256d square = _mm256_mul_pd(diff, diff);
        const __m256d maskedSquare = _mm256_andnot_pd(_mm256_castsi256_pd(gapMask), square);
        accDist.Add(maskedSquare);
      }
      const double entropyHik = -accVect.GetFullSum();
      pAnsMet[k]._entropy.SetValue(entropyHik);
      //Store without computing the square root, therefore it's a square of distance
      pAnsMet[k]._distance.SetValue(accDist.GetFullSum());
    }
    const double totW = accTotW.Get().GetValue();
    //TODO: investigate why this happens
    if (std::fabs(totW - 1.0) > 1e-3) {
      LOCLOG(Warning) << SR_FILE_LINE "The sum of answer weights is " << totW;
    }
    SRAccumVectDbl256 accAvgH; // average entropy over all answer options
    SRAccumVectDbl256 accAvgD;// average distance over all answer options
    const TPqaId nAnswerVects = nAnswers >> SRSimd::_cLogNComps64;
    const TPqaId nVectorized = (nAnswerVects << SRSimd::_cLogNComps64);
#define EASY_SET(metricVar, baseVar) _mm256_set_pd(pAnsMet[baseVar+3].metricVar.GetValue(), \
  pAnsMet[baseVar+2].metricVar.GetValue(), pAnsMet[baseVar + 1].metricVar.GetValue(), \
  pAnsMet[baseVar].metricVar.GetValue())
    for (TPqaId k = 0; k < nVectorized; k += SRSimd::_cLogNComps64) {
      const __m256d curW = EASY_SET(_weight, k);
      const __m256d curH = EASY_SET(_entropy, k);
      const __m256d curD = EASY_SET(_distance, k);
      const __m256d weightedEntropy = _mm256_mul_pd(curW, curH);
      const __m256d weightedDistance = _mm256_mul_pd(curW, curD);
      accAvgH.Add(weightedEntropy);
      accAvgD.Add(weightedDistance);
    }
    for (TPqaId k = nVectorized; k < nAnswers; k++) {
      const __m128d weight = _mm_set1_pd(pAnsMet[k]._weight.GetValue());
      const __m128d metrics = _mm_set_pd(pAnsMet[k]._distance.GetValue(), pAnsMet[k]._entropy.GetValue());
      const __m128d product = _mm_mul_pd(weight, metrics);
      accAvgH.Add(SRVectCompCount(k - nVectorized), product.m128d_f64[0]);
      accAvgD.Add(SRVectCompCount(k - nVectorized), product.m128d_f64[1]);
    }

    // The average entropy over all answers for this question
    const double avgH = accAvgH.GetFullSum() / totW;
    const double nExpectedTargets = std::exp2(avgH);

    // The average of the square of distance over all answers for this question.
    const double avgD = accAvgD.GetFullSum() / totW;

    const double eps = 1e-9;
    const double priority = avgD / ((nExpectedTargets <= eps) ? eps : nExpectedTargets);
    accRunLength.Add(SRDoubleNumber(priority * priority * priority));
    task._pRunLength[i] = accRunLength.Get();
  }
}

} // namespace ProbQA
