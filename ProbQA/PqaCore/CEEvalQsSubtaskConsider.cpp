// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CEEvalQsSubtaskConsider.h"
#include "../PqaCore/CEEvalQsTask.h"
#include "../PqaCore/CEQuiz.h"

using namespace SRPlat;

namespace ProbQA {

template<typename taNumber> TPqaId CEEvalQsSubtaskConsider<taNumber>::CalcPairDistTriangleItems(const TPqaId nAnswers) {
  return (nAnswers * (nAnswers - 1)) >> 1;
}

template<typename taNumber> size_t CEEvalQsSubtaskConsider<taNumber>::CalcPairDistTriangleBytes(const TPqaId nAnswers) {
  return CalcPairDistTriangleItems(nAnswers) * _cAccumVectSize;
}

template class CEEvalQsSubtaskConsider<SRDoubleNumber>;

#define LOCLOG(severityVar) SRLogStream(ISRLogger::Severity::severityVar, engine.GetLogger())

template<> const size_t CEEvalQsSubtaskConsider<SRDoubleNumber>::_cAccumVectSize = sizeof(SRAccumVectDbl256);

namespace {

class ContextDouble {
  __m256d *PTR_RESTRICT const _pPosteriors;
  SRAccumVectDbl256 *PTR_RESTRICT const _pPairDist;

  TPqaId CalcAccCount() const {
    return CEEvalQsSubtaskConsider<SRDoubleNumber>::CalcPairDistTriangleItems(_nAnswers);
  }
public:
  const TPqaId _nTargVects;
  const TPqaId _nAnswers;

public:
  explicit ContextDouble(__m256d *PTR_RESTRICT pPosteriors, SRAccumVectDbl256 *pPairDist, const TPqaId nAnswers,
    const TPqaId nTargVects) : _pPosteriors(pPosteriors), _pPairDist(pPairDist), _nAnswers(nAnswers),
    _nTargVects(nTargVects)
  {
    const TPqaId nAccs = CalcAccCount();
    for (TPqaId i = 0; i < nAccs; i++) {
      new(static_cast<void*>(_pPairDist + i)) SRAccumVectDbl256();
    }
  }
  ~ContextDouble() {
    const TPqaId nAccs = CalcAccCount();
    for (TPqaId i = 0; i < nAccs; i++) {
      _pPairDist[i].~SRAccumVectDbl256();
    }
  }

  void ResetAccs() {
    const TPqaId nAccs = CalcAccCount();
    for (TPqaId i = 0; i < nAccs; i++) {
      _pPairDist[i].Reset();
    }
  }

  ATTR_NOALIAS __m256d &PTR_RESTRICT ModPosterior(const TPqaId iAnswer, const TPqaId iTargVect) {
    return _pPosteriors[iAnswer*_nTargVects + iTargVect];
  }

  ATTR_NOALIAS SRAccumVectDbl256 &PTR_RESTRICT ModPairDist(const TPqaId iSmAns, const TPqaId iGtAns) {
    assert(iSmAns < iGtAns);
    assert(iGtAns >= 1);
    return _pPairDist[CEEvalQsSubtaskConsider<SRDoubleNumber>::CalcPairDistTriangleItems(iGtAns) + iSmAns];
  }
};

} // anonymous namespace

template<> void CEEvalQsSubtaskConsider<SRDoubleNumber>::Run() {
  auto &PTR_RESTRICT task = static_cast<const TTask&>(*GetTask());
  auto &PTR_RESTRICT engine = static_cast<const CpuEngine<SRDoubleNumber>&>(task.GetBaseEngine());
  const CEQuiz<SRDoubleNumber> &PTR_RESTRICT quiz = task.GetQuiz();
  const TPqaId nAnswers = engine.GetDims()._nAnswers;
  const TPqaId nTargVects = SRMath::RShiftRoundUp(engine.GetDims()._nTargets, SRSimd::_cLogNComps64);
  auto *PTR_RESTRICT pPriors = SRCast::CPtr<__m256d>(quiz.GetPriorMants());
  auto *PTR_RESTRICT pAnsMets = SR_STACK_ALLOC(AnswerMetrics<SRDoubleNumber>, nAnswers);
  ContextDouble ctx(SR_STACK_ALLOC_ALIGN(__m256d, nTargVects * nAnswers),
    SR_STACK_ALLOC_ALIGN(SRAccumVectDbl256, CalcPairDistTriangleItems(nAnswers)), nAnswers, nTargVects);

  SRAccumulator<SRDoubleNumber> accRunLength(SRDoubleNumber(0.0));
  for (TPqaId i = _iFirst; i < _iLimit; i++) {
    if (engine.GetQuestionGaps().IsGap(i) || SRBitHelper::Test(quiz.GetQAsked(), i)) {
      // Set 0 probability to this question
      task._pRunLength[i] = accRunLength.Get();
      continue;
    }
    ctx.ResetAccs();
    SRAccumulator<SRDoubleNumber> accTotW(SRDoubleNumber(0.0));
    for (TPqaId k = 0; k < ctx._nAnswers; k++) {
      SRAccumVectDbl256 accVect;
      const __m256d *psAik = SRCast::CPtr<__m256d>(&engine.GetA(i, k, 0));
      const __m256d *pmDi = SRCast::CPtr<__m256d>(&engine.GetD(i, 0));
      for (TPqaId j = 0; j < ctx._nTargVects; j++) {
        const __m256d Pr_Qi_eq_k_given_Tj = _mm256_div_pd(SRSimd::Load<false>(psAik+j), SRSimd::Load<false>(pmDi+j));
        const __m256d likelihood = _mm256_mul_pd(Pr_Qi_eq_k_given_Tj, SRSimd::Load<true>(pPriors + j));
        const uint8_t gaps = engine.GetTargetGaps().GetQuad(j);
        const __m256i gapMask = SRSimd::SetToBitQuadHot(gaps);
        const __m256d maskedLH = _mm256_andnot_pd(_mm256_castsi256_pd(gapMask), likelihood);
        SRSimd::Store<true>(&(ctx.ModPosterior(k, j)), maskedLH);
        //TODO: profiler shows this as the bottleneck (23%)
        accVect.Add(maskedLH);
      }
      const double Wk = accVect.PreciseSum();
      accTotW.Add(SRDoubleNumber::FromDouble(Wk));
      pAnsMets[k]._weight.SetValue(Wk);
      const __m256d vWk = _mm256_set1_pd(Wk);

      accVect.Reset(); // reuse for entropy summation
      const bool bSavePosterior = (k + 1 != ctx._nAnswers);
      for (TPqaId j = 0; j < ctx._nTargVects; j++) {
        // So far there are likelihoods stored, rather than probabilities
        __m256d *pPostLH = &(ctx.ModPosterior(k, j));
        // Normalize to probabilities
        const __m256d posteriors = _mm256_div_pd(SRSimd::Load<true>(pPostLH), vWk);
        // Store for computing distances in the passes over subsequent answers (if current answer is not the last)
        if (bSavePosterior) { //TODO: confirm proper branch prediction here
          SRSimd::Store<true>(pPostLH, posteriors);
        }

        //TODO: remove if entropy is not needed
        //const uint8_t gaps = engine.GetTargetGaps().GetQuad(j);
        //const __m256i gapMask = SRSimd::SetToBitQuadHot(gaps);

        // Calculate negated entropy component: negated self-information multiplied by probability of its event.
        //const __m256d Hikj = _mm256_mul_pd(posteriors, SRVectMath::Log2Hot(posteriors));
        //const __m256d maskedHikj = _mm256_andnot_pd(_mm256_castsi256_pd(gapMask), Hikj);
        //accVect.Add(maskedHikj);

        for (TPqaId t = 0; t < k; t++) {
          // Gaps must have been masked out
          const __m256d *PTR_RESTRICT pFellowPost = &(ctx.ModPosterior(t, j));
          const __m256d fellowPost = SRSimd::Load<true>(pFellowPost);
          const __m256d diff = _mm256_sub_pd(posteriors, fellowPost);
          const __m256d square = _mm256_mul_pd(diff, diff);
          //TODO: profiler shows this as the bottleneck (42%)
          ctx.ModPairDist(t, k).Add(square);
        }
      }
      const double entropyHik = -accVect.PreciseSum();
      pAnsMets[k]._entropy.SetValue(entropyHik);
    }
    const double totW = accTotW.Get().GetValue();
    if (std::fabs(totW - 1.0) > 1e-3) {
      LOCLOG(Warning) << SR_FILE_LINE "The sum of answer weights is " << totW;
    }

    //TODO: vectorize - calculate accumulator sums in quads, and calculate 4 square roots at once
    for (TPqaId k = 0; k < ctx._nAnswers; k++) {
      SRAccumulator<SRDoubleNumber> accD(SRDoubleNumber(0.0));
      for (TPqaId t = 0; t < k; t++) {
        const double dist2 = ctx.ModPairDist(t, k).PreciseSum();
        if (dist2 > 2) {
          printf("x%lfx", dist2);
        }
        const double dist = std::sqrt(dist2);
        accD.Add(SRDoubleNumber::FromDouble(dist * pAnsMets[t]._weight.GetValue()));
      }
      for (TPqaId t = k + 1; t < ctx._nAnswers; t++) {
        const double dist2 = ctx.ModPairDist(k, t).PreciseSum();
        if (dist2 > 2) {
          printf("x%lfx", dist2);
        }
        const double dist = std::sqrt(dist2);
        accD.Add(SRDoubleNumber::FromDouble(dist * pAnsMets[t]._weight.GetValue()));
      }
      //TODO: any normalization by totW here?
      pAnsMets[k]._distance.SetValue(accD.Get().GetValue());
    }

    SRAccumVectDbl256 accAvgH; // average entropy over all answer options
    SRAccumVectDbl256 accAvgD;// average distance over all answer options
    const TPqaId nAnswerVects = (ctx._nAnswers >> SRSimd::_cLogNComps64);
    const TPqaId nVectorized = (nAnswerVects << SRSimd::_cLogNComps64);

#define EASY_SET(metricVar, baseVar) _mm256_set_pd(pAnsMets[baseVar+3].metricVar.GetValue(), \
  pAnsMets[baseVar+2].metricVar.GetValue(), pAnsMets[baseVar + 1].metricVar.GetValue(), \
  pAnsMets[baseVar].metricVar.GetValue())

    for (TPqaId k = 0; k < nVectorized; k += SRSimd::_cNComps64) {
      const __m256d curD = EASY_SET(_distance, k);
      const __m256d curW = EASY_SET(_weight, k);
      //const __m256d curH = EASY_SET(_entropy, k);
      //const __m256d weightedEntropy = _mm256_mul_pd(curW, curH);
      const __m256d weightedDistance = _mm256_mul_pd(curW, curD);
      //accAvgH.Add(weightedEntropy);
      accAvgD.Add(weightedDistance);
    }

#undef EASY_SET

    for (TPqaId k = nVectorized; k < ctx._nAnswers; k++) {
      const __m128d weight = _mm_set1_pd(pAnsMets[k]._weight.GetValue());
      const __m128d metrics = _mm_set_pd(pAnsMets[k]._distance.GetValue(), pAnsMets[k]._entropy.GetValue());
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
    constexpr double cMaxD = SRMath::_cSqrt2;
    constexpr double cExpMul = cMaxExp / cMaxD;

    // The average entropy over all answers for this question
    const double avgH = averages.m128d_f64[0];
    const double nExpectedTargets = std::exp2(avgH);
    constexpr double epsET = 1.0;
    const double stableET = ((nExpectedTargets <= epsET) ? epsET : nExpectedTargets);

    // The average of the square of distance over all answers for this question.
    const double avgD = averages.m128d_f64[1];
    if (avgD > cMaxD) {
      printf("X%lfX", avgD);
      LOCLOG(Warning) << SR_FILE_LINE "Got avgD=" << avgD;
    }
    //const double stableDist = std::sqrt(avgD);
    constexpr double epsD = 1e-200;
    const double stableDist = ((avgD <= epsD) ? epsD : avgD);

    const double priority = std::expm1(cExpMul * stableDist /* / stableET */);
    accRunLength.Add(SRDoubleNumber::FromDouble(priority));

    task._pRunLength[i] = accRunLength.Get();
  }
  //TODO: perhaps check task._pRunLength[_iLimit-1] for overflow/underflow instead of CpuEngine::NextQuestion()
}

} // namespace ProbQA
