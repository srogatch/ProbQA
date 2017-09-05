// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CENormPriorsSubtaskMax.h"
#include "../PqaCore/CENormPriorsTask.h"
#include "../PqaCore/DoubleNumber.h"

using namespace SRPlat;

namespace ProbQA {

template<typename taNumber> CENormPriorsSubtaskMax<taNumber>::CENormPriorsSubtaskMax(CENormPriorsTask<taNumber> *pTask,
  const TPqaId iFirstVT, const TPqaId iLimVT) : SRBaseSubtask(pTask), _iFirstVT(iFirstVT), _iLimVT(iLimVT)
{ }

template class CENormPriorsSubtaskMax<DoubleNumber>;

} // namespace ProbQA
