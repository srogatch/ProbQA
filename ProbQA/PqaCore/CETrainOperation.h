// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/PqaCommon.h"
#include "../PqaCore/CpuEngine.fwd.h"
#include "../PqaCore/CETrainTaskNumSpec.h"

namespace ProbQA {

template<typename taNumber> class CETrainOperation {
  CpuEngine<taNumber> &_engine;
  const CETrainTaskNumSpec<taNumber>& _numSpec;
  const TPqaId _iTarget;

  // A method for taNumber=SRDoubleNumber only. Overload for other taNumber values.
  void ProcessOne(const AnsweredQuestion& aq, const double twoB, const double bSquare);

public:
  CETrainOperation(CpuEngine<taNumber> &engine, const TPqaId iTarget, const CETrainTaskNumSpec<taNumber>& numSpec)
    : _engine(engine), _iTarget(iTarget), _numSpec(numSpec) { }

  // Inputs must have been verified. Maintenance switch and reader-writer sync must be locked.
  void Perform2(const AnsweredQuestion& aqFirst, const AnsweredQuestion& aqSecond);
  void Perform1(const AnsweredQuestion& aq);
};

} // namespace ProbQA
