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
        accVect.Add(maskedLH);
      }
      const double Wk = accVect.PreciseSum();
      accTotW.Add(SRDoubleNumber::FromDouble(Wk));
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
        // Operations should be faster if components are zero, so zero them out early.
        const __m256d maskedDiff = _mm256_andnot_pd(_mm256_castsi256_pd(gapMask), diff);
        const __m256d square = _mm256_mul_pd(maskedDiff, maskedDiff);
        accDist.Add(square);
      }
      double sumDist2;
      const double entropyHik = -accVect.PairSum(accDist, sumDist2);
      pAnsMet[k]._entropy.SetValue(entropyHik);
      //Store without computing the square root, therefore it's a square of distance
      pAnsMet[k]._distance.SetValue(sumDist2);
    }
    const double totW = accTotW.Get().GetValue();
    //TODO: investigate why this happens
    if (std::fabs(totW - 1.0) > 1e-3) {
      LOCLOG(Warning) << SR_FILE_LINE "The sum of answer weights is " << totW;
    }

    SRAccumVectDbl256 accAvgH; // average entropy over all answer options
    SRAccumVectDbl256 accAvgD;// average distance over all answer options
    const TPqaId nAnswerVects = (nAnswers >> SRSimd::_cLogNComps64);
    const TPqaId nVectorized = (nAnswerVects << SRSimd::_cLogNComps64);

#define EASY_SET(metricVar, baseVar) _mm256_set_pd(pAnsMet[baseVar+3].metricVar.GetValue(), \
  pAnsMet[baseVar+2].metricVar.GetValue(), pAnsMet[baseVar + 1].metricVar.GetValue(), \
  pAnsMet[baseVar].metricVar.GetValue())

    for (TPqaId k = 0; k < nVectorized; k += SRSimd::_cNComps64) {
      const __m256d curD2 = EASY_SET(_distance, k);
      const __m256d curW = EASY_SET(_weight, k);
      const __m256d curH = EASY_SET(_entropy, k);
      // Because pAnsMet[k]._distance contains the square distance, compute the square root here.
      const __m256d curD = _mm256_sqrt_pd(curD2);
      const __m256d weightedEntropy = _mm256_mul_pd(curW, curH);
      const __m256d weightedDistance = _mm256_mul_pd(curW, curD);
      accAvgH.Add(weightedEntropy);
      accAvgD.Add(weightedDistance);
    }

#undef EASY_SET

    for (TPqaId k = nVectorized; k < nAnswers; k++) {
      const __m128d weight = _mm_set1_pd(pAnsMet[k]._weight.GetValue());
      // Because pAnsMet[k]._distance contains the square distance, compute the square root here.
      const double distance = std::sqrt(pAnsMet[k]._distance.GetValue());
      const __m128d metrics = _mm_set_pd(distance, pAnsMet[k]._entropy.GetValue());
      const __m128d product = _mm_mul_pd(weight, metrics);
      //TODO: vectorize
      accAvgH.Add(SRVectCompCount(k - nVectorized), product.m128d_f64[0]);
      accAvgD.Add(SRVectCompCount(k - nVectorized), product.m128d_f64[1]);
    }

    __m128d averages;
    averages.m128d_f64[0] = accAvgH.PairSum(accAvgD, averages.m128d_f64[1]);
    const __m128d vTotW = _mm_set1_pd(totW);
    averages = _mm_div_pd(averages, vTotW);

    constexpr double cMaxExp = 664; //Don't call exp(x) for x>cMaxExp
    constexpr double cSqrt2 = 1.4142135623730950488016887242097;
    // Divide by std::sqrt(2.0) if using first-degree distance.
    constexpr double cExpMul = cMaxExp / cSqrt2;

    // The average entropy over all answers for this question
    const double avgH = averages.m128d_f64[0];
    const double nExpectedTargets = std::exp2(avgH);
    constexpr double epsET = 1.0; // 1e-9
    const double stableET = ((nExpectedTargets <= epsET) ? epsET : nExpectedTargets);

    // The average of the square of distance over all answers for this question.
    const double avgD = averages.m128d_f64[1];
    if (avgD > cSqrt2) {
      LOCLOG(Warning) << SR_FILE_LINE "Got avgD=" << avgD;
    }
    constexpr double epsD = 1e-200;
    const double stableDist = ((avgD <= epsD) ? epsD : avgD);
    //const double scaledDist = avgD * 1e20;
    //constexpr double epsD = 1e-20;
    //const double stableDist = ((scaledDist <= epsD) ? epsD : scaledDist);
    //const double squareDist = stableDist * stableDist;

    //FIXME: growth of distance polynomial degree has the opposite effects for D<1 and D>1.
    //TODO: devise a function which has consistent effects
    //const double priority = squareDist * squareDist / stableET * stableET;
    //accRunLength.Add(SRDoubleNumber::FromDouble(priority * priority * priority));

    const double priority = std::expm1(cExpMul * stableDist / stableET);
    accRunLength.Add(SRDoubleNumber::FromDouble(priority));

    task._pRunLength[i] = accRunLength.Get();
  }
  //TODO: perhaps check task._pRunLength[_iLimit-1] for overflow/underflow instead of CpuEngine::NextQuestion()
}

} // namespace ProbQA
