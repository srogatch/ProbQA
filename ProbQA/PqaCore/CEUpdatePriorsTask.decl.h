// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CEUpdatePriorsTask.fwd.h"
#include "../PqaCore/CEQuiz.fwd.h"
#include "../PqaCore/CpuEngine.fwd.h"
#include "../PqaCore/CEBaseTask.h"

namespace ProbQA {

template<typename taNumber> class CEUpdatePriorsTask : public CEBaseTask {
public: // variables
  const CEQuiz<taNumber> *const _pQuiz;
  const AnsweredQuestion* const _pAQs;
  const TPqaId _nAnswered;
  const uint32_t _nVectsInCache;

public: // methods
  CEUpdatePriorsTask(CpuEngine<taNumber> &engine, CEQuiz<taNumber> &quiz, const TPqaId nAnswered,
    const AnsweredQuestion* const pAQs, const uint32_t nVectsInCache);
};

} // namespace ProbQA
