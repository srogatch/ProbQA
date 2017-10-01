// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CEQuiz.fwd.h"
#include "../PqaCore/CpuEngine.fwd.h"
#include "../PqaCore/Interface/PqaCommon.h"
#include "../PqaCore/RatingsHeap.h"
#include "../PqaCore/PqaRange.h"

namespace ProbQA {

template<typename taNumber> class CERadixSortRatingsTask : public CEBaseTask {
  const CEQuiz<taNumber> *const _pQuiz;
  RatedTarget *_pRatings;
  RatedTarget *_pTempRatings;
  TPqaId *_pCounters;
  TPqaId *_pOffsets;

public:
  explicit CERadixSortRatingsTask(CpuEngine<taNumber> &engine, const CEQuiz<taNumber> &quiz, RatedTarget *pRatings,
    RatedTarget *pTempRatings, TPqaId *pCounters, TPqaId *pOffsets) : CEBaseTask(engine), _pQuiz(&quiz),
    _pRatings(pRatings), _pTempRatings(pTempRatings), _pCounters(pCounters), _pOffsets(pOffsets) { }

  const CEQuiz<taNumber>& GetQuiz() const { return *_pQuiz; }
  RatedTarget* ModRatings() const { return _pRatings; }
  RatedTarget* ModTempRatings() const { return _pTempRatings; }
  TPqaId* ModCounters() const { return _pCounters; }
  TPqaId* ModOffsets() const { return _pOffsets; }
};

} // namespace ProbQA
