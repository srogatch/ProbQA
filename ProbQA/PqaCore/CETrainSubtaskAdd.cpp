// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CETrainSubtaskAdd.h"
#include "../PqaCore/CETrainTask.h"
#include "../PqaCore/CpuEngine.h"
#include "../PqaCore/CETrainOperation.h"

using namespace SRPlat;

namespace ProbQA {

template class CETrainSubtaskAdd<SRDoubleNumber>;

template<> void CETrainSubtaskAdd<SRDoubleNumber>::Run() {
  auto& cTask = static_cast<const TTask&>(*GetTask()); // enable optimizations with const
  auto& engine = static_cast<CpuEngine<SRDoubleNumber>&>(cTask.GetBaseEngine());
  TPqaId iLast = cTask._last[_iWorker];
  if (iLast == cInvalidPqaId) {
    return;
  }
  const TPqaId *const cPrev = cTask._prev;

  CETrainOperation<SRDoubleNumber> trainOp(engine, cTask._iTarget, cTask._numSpec);
  do {
    const AnsweredQuestion& aqFirst = cTask._pAQs[iLast];
    iLast = cPrev[iLast];
    if (iLast == cInvalidPqaId) {
      trainOp.Perform1(aqFirst);
      return;
    }
    const AnsweredQuestion& aqSecond = cTask._pAQs[iLast];
    trainOp.Perform2(aqFirst, aqSecond);
    iLast = cPrev[iLast];
  } while (iLast != cInvalidPqaId);
}

} // namespace ProbQA
