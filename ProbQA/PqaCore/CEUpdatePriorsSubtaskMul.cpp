// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CEUpdatePriorsSubtaskMul.h"
#include "../PqaCore/CEUpdatePriorsTask.h"
#include "../PqaCore/DoubleNumber.h"

using namespace SRPlat;

namespace ProbQA {

template<typename taNumber> CEUpdatePriorsSubtaskMul<taNumber>::CEUpdatePriorsSubtaskMul(
  CEUpdatePriorsTask<taNumber> *pTask, const TPqaId iFirstTarget, const TPqaId iLimTarget)
  : SRBaseSubtask(pTask), _iFirstTarget(iFirstTarget), _iLimTarget(iLimTarget)
{ }

template<> void CEUpdatePriorsSubtaskMul<DoubleNumber>::Run() {

}

template class CEUpdatePriorsSubtaskMul<DoubleNumber>;

} // namespace ProbQA
