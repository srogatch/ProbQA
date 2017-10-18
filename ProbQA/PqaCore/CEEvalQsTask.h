// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CEEvalQsTask.fwd.h"
#include "../PqaCore/CEQuiz.fwd.h"
#include "../PqaCore/CpuEngine.fwd.h"
#include "../PqaCore/CEBaseTask.h"
#include "../PqaCore/AnswerMetrics.h"

namespace ProbQA {

template<typename taNumber> class CEEvalQsSubtaskConsider;

template<typename taNumber> class CEEvalQsTask : public CEBaseTask {
  friend class CEEvalQsSubtaskConsider<taNumber>;

  const CEQuiz<taNumber> *const _pQuiz;
  taNumber *const _pRunLength;
  const TPqaId _nValidTargets;

public: // methods
  explicit inline CEEvalQsTask(CpuEngine<taNumber> &engine, const CEQuiz<taNumber> &quiz, const TPqaId nValidTargets,
    taNumber *pRunLength)
    : CEBaseTask(engine), _pQuiz(&quiz), _nValidTargets(nValidTargets), _pRunLength(pRunLength)
  { }

  const CEQuiz<taNumber>& GetQuiz() const { return *_pQuiz; }
  const taNumber* GetRunLength() const { return _pRunLength; }
};

} // namespace ProbQA
