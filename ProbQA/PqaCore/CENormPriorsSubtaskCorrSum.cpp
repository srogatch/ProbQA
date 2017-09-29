// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CENormPriorsSubtaskCorrSum.h"
#include "../PqaCore/CENormPriorsTask.h"
#include "../PqaCore/CpuEngine.h"
#include "../PqaCore/CENormPriorsTask.h"
#include "../PqaCore/CEQuiz.h"

using namespace SRPlat;

namespace ProbQA {

template class CENormPriorsSubtaskCorrSum<SRDoubleNumber>;

namespace {

struct ContextDouble {
  const GapTracker<TPqaId> *PTR_RESTRICT _pGt;
  const CENormPriorsTask<SRDoubleNumber> *PTR_RESTRICT _pTask;
  __m256d *PTR_RESTRICT _pMants;
  __m256i *PTR_RESTRICT _pExps;

  // Returns the addend for bucket summator
  ATTR_NOALIAS inline __m256d __vectorcall Process(const TPqaId iVect, __m256i &PTR_RESTRICT exps) {
    const __m256d oldMants = SRSimd::Load<false, __m256d>(_pMants + iVect);
    const __m256i origExps = _mm256_add_epi64(SRSimd::Load<false, __m256i>(_pExps + iVect),
      SRSimd::ExtractExponents64<false>(oldMants));
    const uint8_t gaps = _pGt->GetQuad(iVect);
    const __m256i normExps = _mm256_add_epi64(origExps, _pTask->_corrExp);
    // Avoid subnormal numbers (pretend they are zeros)
    const __m256i isExpBelow1 = _mm256_cmpgt_epi64(_mm256_set1_epi64x(1), normExps);
    const __m256i assume0 = _mm256_or_si256(isExpBelow1, SRSimd::SetToBitQuadHot(gaps));

    const __m256d newMants = _mm256_andnot_pd(_mm256_castsi256_pd(assume0),
      SRSimd::ReplaceExponents(oldMants, normExps));
    SRSimd::Store<false>(_pExps + iVect, _mm256_setzero_si256());
    SRSimd::Store<false>(_pMants + iVect, newMants);
    exps = _mm256_andnot_si256(assume0, normExps);
    return newMants;
  }
};

}

template<> void CENormPriorsSubtaskCorrSum<SRDoubleNumber>::Run() {
  ContextDouble ctx;
  ctx._pTask = static_cast<const TTask*>(GetTask());
  auto &PTR_RESTRICT engine = static_cast<const CpuEngine<SRDoubleNumber>&>(ctx._pTask->GetBaseEngine());
  const CEQuiz<SRDoubleNumber> &PTR_RESTRICT quiz = ctx._pTask->GetQuiz();
  SRBucketSummatorPar<SRDoubleNumber> &PTR_RESTRICT bsp = (ctx._pTask->GetBSP());
  ctx._pGt = &engine.GetTargetGaps();
  ctx._pExps = SRCast::Ptr<__m256i>(quiz.GetTlhExps());
  ctx._pMants = SRCast::Ptr<__m256d>(quiz.GetPriorMants());

  bsp.ZeroBuckets(_iWorker);

  for (TPqaId i = _iFirst, iEn = _iLimit; i < iEn; i++) {
    __m256i exps;
    const __m256d addend = ctx.Process(i, exps);
    bsp.Add(_iWorker, addend, exps);
  }
  _mm_sfence();
}

} // namespace ProbQA
