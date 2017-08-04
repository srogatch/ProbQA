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
  CEQuiz<taNumber> *_pQuiz;
  const AnsweredQuestion* const _pAQs;
  const TPqaId _nAnswered;

public:
  CEUpdatePriorsTask(CpuEngine<taNumber> *pCe, CEQuiz<taNumber> *pQuiz, const TPqaId nAnswered,
    const AnsweredQuestion* const pAQs);
};

} // namespace ProbQA
