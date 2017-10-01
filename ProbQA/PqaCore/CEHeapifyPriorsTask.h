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

template<typename taNumber> class CEHeapifyPriorsTask : public CEBaseTask {
  const CEQuiz<taNumber> *const _pQuiz;
  RatedTarget *_pRatings;
  TPqaId *_pPieceLimits;

public:
  explicit CEHeapifyPriorsTask(CpuEngine<taNumber> &engine, const CEQuiz<taNumber> &quiz, RatedTarget *pRatings,
    TPqaId *pPieceLimits) : CEBaseTask(engine), _pQuiz(&quiz), _pRatings(pRatings), _pPieceLimits(pPieceLimits) { }

  const CEQuiz<taNumber>& GetQuiz() const { return *_pQuiz; }
  RatedTarget* ModRatings() const { return _pRatings; }
  TPqaId* ModPieceLimits() const { return _pPieceLimits; }
};

} // namespace ProbQA
