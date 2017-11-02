// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CEEvalQsSubtaskConsider.h"
#include "../PqaCore/CEEvalQsTask.h"
#include "../PqaCore/CEQuiz.h"

using namespace SRPlat;

namespace ProbQA {

template<typename taNumber> size_t CEEvalQsSubtaskConsider<taNumber>::CalcStackReq(const EngineDefinition& engDef) {
  return SRSimd::GetPaddedBytes(sizeof(taNumber) * engDef._dims._nTargets) * 2
    + sizeof(AnswerMetrics<taNumber>) * engDef._dims._nAnswers;
}

template class CEEvalQsSubtaskConsider<SRDoubleNumber>;

#define LOCLOG(severityVar) SRLogStream(ISRLogger::Severity::severityVar, engine.GetLogger())

FLOAT_PRECISE_BEGIN
template<> double CEEvalQsSubtaskConsider<SRDoubleNumber>::CalcVelocityComponent(const double V,
  const TPqaId nTargets) {
  // Min exponent : -1023
  // Exponent due to subnormals : -52
  // ln(2**1075) = 1075 * ln(2) = 1075 * 0.6931471805599453 = 745.1332191019411975
  const double lnV = ((V == 0) ? _cLn0Stab : std::log(V));
  const double powT = double(nTargets) * nTargets;
  // The order of operations is important for numerical stability.
  const double vComp = 1 / (_cLnMaxV - lnV + _cLnMaxV / powT);
  return vComp;
}
FLOAT_PRECISE_END

namespace {
  const __m256d gcProbEps = _mm256_set1_pd(std::ldexp(1.0, -960));
}

template<> void CEEvalQsSubtaskConsider<SRDoubleNumber>::Run() {
  auto &PTR_RESTRICT task = static_cast<const TTask&>(*GetTask());
  auto &PTR_RESTRICT engine = static_cast<const CpuEngine<SRDoubleNumber>&>(task.GetBaseEngine());
  const CEQuiz<SRDoubleNumber> &PTR_RESTRICT quiz = task.GetQuiz();
  const TPqaId nAnswers = engine.GetDims()._nAnswers;
  const TPqaId nTargVects = SRMath::RShiftRoundUp(engine.GetDims()._nTargets, SRSimd::_cLogNComps64);
  auto *const PTR_RESTRICT pPriors = SRCast::CPtr<__m256d>(quiz.GetPriorMants());
  auto *const PTR_RESTRICT pAnsMets = SR_STACK_ALLOC(AnswerMetrics<SRDoubleNumber>, nAnswers);
  __m256d *const PTR_RESTRICT pInvDi = SR_STACK_ALLOC_ALIGN(__m256d, nTargVects);
  __m256d *const PTR_RESTRICT pPosteriors = SR_STACK_ALLOC_ALIGN(__m256d, nTargVects);

  SRAccumulator<SRDoubleNumber> accRunLength(SRDoubleNumber(0.0));
  for (TPqaId i = _iFirst; i < _iLimit; i++) {
    if (engine.GetQuestionGaps().IsGap(i) || SRBitHelper::Test(quiz.GetQAsked(), i)) {
      // Set 0 probability to this question
      task._pRunLength[i] = accRunLength.Get();
      continue;
    }
    const __m256d *const PTR_RESTRICT pmDi = SRCast::CPtr<__m256d>(&(engine.GetD(i, 0)));
    SRAccumulator<SRDoubleNumber> accTotW(SRDoubleNumber(0.0));
    SRAccumVectDbl256 accL;
    for (TPqaId k = 0; k < nAnswers; k++) {
      SRAccumVectDbl256 accLhEnt; // For likelihood and entropy
      const __m256d *const PTR_RESTRICT psAik = SRCast::CPtr<__m256d>(&(engine.GetA(i, k, 0)));
      const bool isAns0 = (k == 0);
      for (TPqaId j = 0; j < nTargVects; j++) {
        const uint8_t gaps = engine.GetTargetGaps().GetQuad(j);
        const __m256d gapMask = _mm256_castsi256_pd(SRSimd::SetToBitQuadHot(gaps));
        const __m256d priors = _mm256_andnot_pd(gapMask, SRSimd::Load<true>(pPriors + j));

        __m256d invCountTotal; // mD[i][j]
        if (isAns0) {
          const __m256d vDij = SRSimd::Load<false>(pmDi + j);
          invCountTotal = _mm256_andnot_pd(gapMask, _mm256_div_pd(SRVectMath::_cdOne256, vDij));
          SRSimd::Store<true>(pInvDi + j, invCountTotal);
        }
        else {
          invCountTotal = SRSimd::Load<true>(pInvDi + j);
        }

        const __m256d Pr_Qi_eq_k_given_Tj = _mm256_mul_pd(SRSimd::Load<false>(psAik+j), invCountTotal);
        const __m256d likelihood = _mm256_mul_pd(Pr_Qi_eq_k_given_Tj, priors);
        
        SRSimd::Store<true>(pPosteriors + j, likelihood);
        //TODO: profiler shows this as the bottleneck (23%)
        accLhEnt.Add(likelihood);
      }
      const double Wk = accLhEnt.PreciseSum();
      accTotW.Add(SRDoubleNumber::FromDouble(Wk));
      pAnsMets[k]._weight.SetValue(Wk);
      const __m256d invWk = _mm256_div_pd(SRVectMath::_cdOne256, _mm256_set1_pd(Wk));

      accLhEnt.Reset(); // reuse for entropy summation
      SRAccumVectDbl256 accV; // velocity
      for (TPqaId j = 0; j < nTargVects; j++) {
        // So far there are likelihoods stored, rather than probabilities. Normalize to probabilities.
        const __m256d posteriors = _mm256_mul_pd(SRSimd::Load<true>(pPosteriors+j), invWk);

        const uint8_t gaps = engine.GetTargetGaps().GetQuad(j);
        const __m256d gapMask = _mm256_castsi256_pd(SRSimd::SetToBitQuadHot(gaps));

        // Operations should be faster if components are zero, so zero them out early.
        const __m256d priors = _mm256_andnot_pd(gapMask, SRSimd::Load<true>(pPriors + j));

        // Calculate negated entropy component: negated self-information multiplied by probability of its event.
        const __m256d l2post = _mm256_andnot_pd(gapMask, SRVectMath::Log2Hot(posteriors));
        //DEBUG
        //for (int8_t c = 0; c <= 3; c++) {
        //  if (l2post.m256d_f64[c] > 0) {
        //    __debugbreak();
        //  }
        //}
        const __m256d Hikj = _mm256_mul_pd(posteriors, l2post);
        accLhEnt.Add(Hikj);

        const __m256d invDij = SRSimd::Load<true>(pInvDi + j);
        accL.Add(_mm256_div_pd(_mm256_mul_pd(invDij, invDij), l2post));

        const __m256d diff = _mm256_sub_pd(posteriors, priors);

        //const __m256d priorsGood = _mm256_cmp_pd(priors, gcProbEps, _CMP_GT_OQ);
        //const __m256d ratio = _mm256_div_pd(posteriors, priors);
        //const __m256d diff = _mm256_and_pd(priorsGood, SRVectMath::Log2Hot(ratio));
        //const __m256d absDiff = SRSimd::AbsF64(diff);

        const __m256d square = _mm256_mul_pd(diff, diff);
        accV.Add(square);
      }
      double velocity;
      const double entropyHik = -accLhEnt.PairSum(accV, velocity);
      pAnsMets[k]._entropy.SetValue(entropyHik);
      pAnsMets[k]._velocity.SetValue(velocity);
    }
    const double totW = accTotW.Get().GetValue();
    if (std::fabs(totW - 1.0) > 1e-3) {
      LOCLOG(Warning) << SR_FILE_LINE "The sum of answer weights is " << totW;
    }

    SRAccumVectDbl256 accAvgH; // average entropy over all answer options
    SRAccumVectDbl256 accAvgV;// average velocity over all answer options
    const TPqaId nAnswerVects = (nAnswers >> SRSimd::_cLogNComps64);
    const TPqaId nVectorized = (nAnswerVects << SRSimd::_cLogNComps64);

#define EASY_SET(metricVar, baseVar) _mm256_set_pd(pAnsMets[baseVar+3].metricVar.GetValue(), \
  pAnsMets[baseVar+2].metricVar.GetValue(), pAnsMets[baseVar + 1].metricVar.GetValue(), \
  pAnsMets[baseVar].metricVar.GetValue())

    for (TPqaId k = 0; k < nVectorized; k += SRSimd::_cNComps64) {
      const __m256d curW = EASY_SET(_weight, k);
      
      const __m256d curH = EASY_SET(_entropy, k);
      const __m256d weightedEntropy = _mm256_mul_pd(curW, curH);
      accAvgH.Add(weightedEntropy);

      const __m256d curV2 = EASY_SET(_velocity, k);
      const __m256d curV = _mm256_sqrt_pd(curV2);
      const __m256d weightedVelocity = _mm256_mul_pd(curW, curV);
      accAvgV.Add(weightedVelocity);
    }

#undef EASY_SET

    for (TPqaId k = nVectorized; k < nAnswers; k++) {
      const __m128d weight = _mm_set1_pd(pAnsMets[k]._weight.GetValue());
      const double velocity = std::sqrt(pAnsMets[k]._velocity.GetValue());
      const __m128d metrics = _mm_set_pd(velocity, pAnsMets[k]._entropy.GetValue());
      const __m128d product = _mm_mul_pd(weight, metrics);
      const SRVectCompCount iComp = static_cast<SRVectCompCount>(k - nVectorized);
      //TODO: vectorize
      accAvgH.Add(iComp, product.m128d_f64[0]);
      accAvgV.Add(iComp, product.m128d_f64[1]);
    }

    __m128d averages;
    averages.m128d_f64[0] = accAvgH.PairSum(accAvgV, averages.m128d_f64[1]);
    const __m128d normalizer = _mm_set1_pd(totW);
    averages = _mm_div_pd(averages, normalizer);

    // The average entropy over all answers for this question
    const double avgH = averages.m128d_f64[0];
    const double nExpectedTargets = std::exp2(avgH);
    if (nExpectedTargets + 1e-6 < 1) {
      LOCLOG(Warning) << SR_FILE_LINE "Got nExpectedTargets=" << nExpectedTargets << ", entropy=" << avgH;
    }

    const double avgV = averages.m128d_f64[1];
    if (avgV < 0 || avgV > _cMaxV) {
      LOCLOG(Warning) << SR_FILE_LINE "Got avgV=" << avgV;
    }

    const double vComp = CalcVelocityComponent(avgV, task._nValidTargets+1);
    if (vComp <= 0) {
      LOCLOG(Warning) << SR_FILE_LINE "Got vComp=" << vComp;
    }

    //constexpr double epsV = 1e-30;
    //const double scaledV = avgV / epsV;
    //const double stableV = ((scaledV <= epsV) ? epsV : scaledV);
    //const double vComp = stableV;

    const double lack = -accL.PreciseSum();
    if (lack <= 0) {
      LOCLOG(Warning) << SR_FILE_LINE "Got lack=" << lack;
    }

    //TODO: change to integer powers algorithm after best powers are found experimentally.
    const double priority = std::pow(lack, 1) * std::pow(vComp, 9) * std::pow(nExpectedTargets, -3);

    if (priority <= 0 || !std::isfinite(priority)) {
      LOCLOG(Warning) << SR_FILE_LINE "Got priority=" << priority;
    }
    accRunLength.Add(SRDoubleNumber::FromDouble(priority));

    task._pRunLength[i] = accRunLength.Get(); 
  }
  //TODO: perhaps check task._pRunLength[_iLimit-1] for overflow/underflow instead of CpuEngine::NextQuestion()
}

} // namespace ProbQA
