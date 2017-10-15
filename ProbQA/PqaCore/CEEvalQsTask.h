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
  uint8_t *const _pPosteriors;
  AnswerMetrics<taNumber> *const _pAnswerMetrics;
  size_t _threadPosteriorBytes;
  const TPqaId _nValidTargets;

public: // methods
  explicit inline CEEvalQsTask(CpuEngine<taNumber> &engine, const CEQuiz<taNumber> &quiz, const TPqaId nValidTargets,
    taNumber *pRunLength, uint8_t *const pPosteriors, const size_t threadPosteriorBytes,
    AnswerMetrics<taNumber> *const pAnswerMetrics)
    : CEBaseTask(engine), _pQuiz(&quiz), _nValidTargets(nValidTargets), _pRunLength(pRunLength),
    _pPosteriors(pPosteriors), _threadPosteriorBytes(threadPosteriorBytes), _pAnswerMetrics(pAnswerMetrics)
  { }

  const CEQuiz<taNumber>& GetQuiz() const { return *_pQuiz; }
  const taNumber* GetRunLength() const { return _pRunLength; }
};

} // namespace ProbQA
