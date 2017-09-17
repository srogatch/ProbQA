// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CEEvalQsTask.fwd.h"
#include "../PqaCore/CEQuiz.fwd.h"
#include "../PqaCore/CpuEngine.fwd.h"
#include "../PqaCore/CEBaseTask.h"

namespace ProbQA {

template<typename taNumber> class CEEvalQsTask : public CEBaseTask {
  const CEQuiz<taNumber> *const _pQuiz;
  const TPqaId _nValidTargets;
public: // variables

public: // methods
  explicit inline CEEvalQsTask(CpuEngine<taNumber> &engine, const CEQuiz<taNumber> &quiz, const TPqaId nValidTargets)
    : CEBaseTask(engine), _pQuiz(&quiz), _nValidTargets(nValidTargets) { }
};

} // namespace ProbQA
