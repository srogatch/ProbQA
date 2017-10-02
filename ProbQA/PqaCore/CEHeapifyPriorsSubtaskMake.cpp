// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CEHeapifyPriorsSubtaskMake.h"
#include "../PqaCore/CpuEngine.h"
#include "../PqaCore/CEQuiz.h"

using namespace SRPlat;

namespace ProbQA {

//TODO: because this class is not likely to have specialized methods, to avoid excessive listing of all the supported
//  template arguments here, move the implementation to fwd/decl/h header-only idiom.
template class CEHeapifyPriorsSubtaskMake<SRDoubleNumber>;

template<typename taNumber> struct CEHeapifyPriorsSubtaskMake<taNumber>::Context {
  const TTask *PTR_RESTRICT _pTask;
  const CpuEngine<taNumber> *PTR_RESTRICT _pEngine;
  const CEQuiz<taNumber> *PTR_RESTRICT _pQuiz;
  const taNumber* PTR_RESTRICT _pPriors;
  const GapTracker<TPqaId> *PTR_RESTRICT _pGt;
  RatedTarget * PTR_RESTRICT _pRatings;
  TPqaId _iSelLim;

  explicit Context(SRBaseTask *const PTR_RESTRICT pTask, CEHeapifyPriorsSubtaskMake<taNumber> *pSubtask) {
    _pTask = static_cast<const TTask*>(pTask);
    _pQuiz = &(_pTask->GetQuiz());
    _pPriors = _pQuiz->GetPriorMants();
    _iSelLim = pSubtask->_iFirst;
    {
      const char *PTR_RESTRICT pCacheLine = SRCast::CPtr<char>(_pPriors + _iSelLim);
      _mm_prefetch(pCacheLine, _MM_HINT_NTA);
      _mm_prefetch(pCacheLine + SRCpuInfo::_cacheLineBytes, _MM_HINT_NTA);
    }
    _pEngine = static_cast<const CpuEngine<taNumber>*>(&(_pTask->GetBaseEngine()));
    _pGt = &(_pEngine->GetTargetGaps());
    _pRatings = _pTask->ModRatings();
  }

  void Regard(const TPqaId iTarget) {
    if (_pGt->IsGap(iTarget)) {
      return;
    }
    const TPqaAmount prob = _pPriors[iTarget].ToAmount();
    if (prob <= 0) {
      return;
    }
    _pRatings[_iSelLim]._prob = prob;
    _pRatings[_iSelLim]._iTarget = iTarget;
    _iSelLim++;
  }
};



template<typename taNumber> void CEHeapifyPriorsSubtaskMake<taNumber>::Run() {
  Context ctx(GetTask(), this);

  constexpr uint32_t nBytesAhead = (SRCpuInfo::_cacheLineBytes << 1);
  
#define UNROLL(varOffset, varThreshold) \
  __pragma(warning(push)) \
  __pragma(warning(disable:4127)) /* conditional expression is constant */ \
    if(sizeof(taNumber) > (varThreshold)) { \
  __pragma(warning(pop)) \
      _mm_prefetch(SRCast::CPtr<char>(ctx._pPriors + i + (varOffset)) + nBytesAhead, _MM_HINT_NTA); \
    } \
    ctx.Regard(i+varThreshold);

  //if constexpr(sizeof(taNumber) <= 8) {
  //}

  TPqaId i = _iFirst;
  for (const TPqaId iEn=_iLimit-7; i < iEn; i+=8) {
    _mm_prefetch(SRCast::CPtr<char>(ctx._pPriors + i) + nBytesAhead, _MM_HINT_NTA);
    ctx.Regard(i);
    UNROLL(1, 32);
    UNROLL(2, 16);
    UNROLL(3, 32);
    UNROLL(4, 8);
    UNROLL(5, 32);
    UNROLL(6, 16);
    UNROLL(7, 32);
  }
#undef UNROLL

  for (; i < _iLimit; i++) {
    ctx.Regard(i);
  }

  ctx._pTask->ModPieceLimits()[_iWorker] = ctx._iSelLim;
  std::make_heap(ctx._pRatings, ctx._pRatings + ctx._iSelLim);
}

} // namespace ProbQA
