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

template<typename taNumber> CENormPriorsSubtaskCorrSum<taNumber>::CENormPriorsSubtaskCorrSum(
  CENormPriorsTask<taNumber> *pTask) : SRStandardSubtask(pTask) { }

struct ContextDouble {
  const GapTracker<TPqaId> *PTR_RESTRICT _pGt;
  const CENormPriorsTask<SRDoubleNumber> *PTR_RESTRICT _pTask;
  __m256d *PTR_RESTRICT _pMants;
  __m256i *PTR_RESTRICT _pExps;

  // Returns the addend for bucket summator
  ATTR_NOALIAS inline __m256d __vectorcall Process(const TPqaId iVect, __m256i &PTR_RESTRICT exps) {
    const __m256i vi0 = _mm256_setzero_si256();
    const __m256d oldMants = SRSimd::Load<true, __m256d>(_pMants + iVect);
    const __m256i origExps = _mm256_add_epi64(SRSimd::Load<true, __m256i>(_pExps + iVect),
      SRSimd::ExtractExponents64<false>(oldMants));
    const uint8_t gaps = _pGt->GetQuad(iVect);
    const __m256i normExps = _mm256_add_epi64(origExps, _pTask->_corrExp);
    const __m256i isExpBelow0 = _mm256_cmpgt_epi64(vi0, normExps);
    const __m256i assume0 = _mm256_or_si256(isExpBelow0, SRSimd::SetToBitQuadHot(gaps));

    const __m256d newMants = _mm256_andnot_pd(_mm256_castsi256_pd(assume0),
      SRSimd::ReplaceExponents(oldMants, normExps));
    SRSimd::Store<true>(_pExps + iVect, vi0);
    SRSimd::Store<true>(_pMants + iVect, newMants);
    exps = _mm256_andnot_si256(assume0, normExps);
    return newMants;
  }
};

template<> void CENormPriorsSubtaskCorrSum<SRDoubleNumber>::Run() {

  ContextDouble ctx;
  ctx._pTask = static_cast<const CENormPriorsTask<SRDoubleNumber>*>(GetTask());
  auto const &PTR_RESTRICT engine = static_cast<const CpuEngine<SRDoubleNumber>&>(ctx._pTask->GetBaseEngine());
  SRBucketSummator<SRDoubleNumber> &PTR_RESTRICT bs = *(ctx._pTask->_pBs);
  ctx._pGt = &engine.GetTargetGaps();
  ctx._pExps = SRCast::Ptr<__m256i>(ctx._pTask->_pQuiz->GetTlhExps());
  ctx._pMants = SRCast::Ptr<__m256d>(ctx._pTask->_pQuiz->GetTlhMants());

  const bool isAtPartial = (_iLimit + 1 == ctx._pTask->_iPartial);

  bs.ZeroBuckets(_iWorker);

  for (TPqaId i = _iFirst, iEn = (isAtPartial ? ctx._pTask->_iPartial : _iLimit); i < iEn; i++) {
    __m256i exps;
    const __m256d addend = ctx.Process(i, exps);
    bs.Add(_iWorker, addend, exps);
  }
  if (isAtPartial) {

  }
  //TODO: implement
}


template class CENormPriorsSubtaskCorrSum<SRPlat::SRDoubleNumber>;

} // namespace ProbQA
