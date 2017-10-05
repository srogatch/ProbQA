// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/PqaCommon.h"
#include "../PqaCore/CpuEngine.fwd.h"

namespace ProbQA {

template<typename taNumber> class CETrainOperation {
  CpuEngine<taNumber> &_engine;
  const AnsweredQuestion* const _pAQs;
  const TPqaAmount _amount;
  const TPqaId _nQuestions;
  const TPqaId _iTarget;

public:
  CETrainOperation(CpuEngine<taNumber> &engine, const TPqaId nQuestions, const AnsweredQuestion* const pAQs,
    const TPqaId iTarget, const TPqaAmount amount) : _engine(engine), _nQuestions(nQuestions), _pAQs(pAQs),
    _iTarget(iTarget), _amount(amount) { }

  // Inputs must have been verified. Maintenance switch must be locked, but not reader-writer sync.
  void Perform();
};

} // namespace ProbQA
