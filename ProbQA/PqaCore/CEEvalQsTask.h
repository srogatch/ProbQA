// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CEEvalQsTask.fwd.h"
#include "../PqaCore/CEQuiz.fwd.h"
#include "../PqaCore/CpuEngine.fwd.h"
#include "../PqaCore/CEBaseTask.h"

namespace ProbQA {

template<typename taNumber> class CEEvalQsSubtaskConsider;

template<typename taNumber> class CEEvalQsTask : public CEBaseTask {
  friend class CEEvalQsSubtaskConsider<taNumber>;

  const CEQuiz<taNumber> *const _pQuiz;
  uint8_t *const _pBSes;
  taNumber *const _pRunLength;
  uint8_t *const _pPosteriors;
  size_t _threadPosteriorBytes;
  const TPqaId _nValidTargets;
public: // variables

public: // methods
  explicit inline CEEvalQsTask(CpuEngine<taNumber> &engine, const CEQuiz<taNumber> &quiz, const TPqaId nValidTargets,
    uint8_t *pBSes, taNumber *pRunLength, uint8_t *const pPosteriors, const size_t threadPosteriorBytes)
    : CEBaseTask(engine), _pQuiz(&quiz), _nValidTargets(nValidTargets), _pBSes(pBSes), _pRunLength(pRunLength),
    _pPosteriors(pPosteriors), _threadPosteriorBytes(threadPosteriorBytes)
  { }

  const CEQuiz<taNumber>& GetQuiz() const { return *_pQuiz; }
};

} // namespace ProbQA
