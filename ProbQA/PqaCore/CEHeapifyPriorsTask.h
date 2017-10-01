// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/PqaCommon.h"
#include "../PqaCore/RatingsHeap.h"
#include "../PqaCore/PqaRange.h"

namespace ProbQA {

template<typename taNumber> class CEHeapifyPriorsTask : public CEBaseTask {
  const CEQuiz<taNumber> *const _pQuiz;
  RatedTarget *_pRatings;
  PqaRange *_pSourceInfos;

public:
  explicit CEHeapifyPriorsTask(CpuEngine<taNumber> &engine, const CEQuiz<taNumber> &quiz, RatedTarget *pRatings,
    PqaRange *pSourceInfos) : CEBaseTask(engine), _pQuiz(&quiz), _pRatings(pRatings),
    _pSourceInfos(pSourceInfos) { }

  const CEQuiz<taNumber>& GetQuiz() const { return *_pQuiz; }
};

} // namespace ProbQA