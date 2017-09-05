// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CENormPriorsTask.fwd.h"
#include "../PqaCore/Interface/PqaCommon.h"

namespace ProbQA {

template<typename taNumber> class CENormPriorsSubtaskMax : public SRPlat::SRBaseSubtask {
  const TPqaId _iFirstVT; // first vector of targets
  const TPqaId _iLimVT; // limit vector of targets

public:
  CENormPriorsSubtaskMax(CENormPriorsTask<taNumber> *pTask, const TPqaId iFirstVT, const TPqaId iLimVT);
};

} // namespace ProbQA
