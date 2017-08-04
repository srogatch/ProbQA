// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CEUpdatePriorsTask.fwd.h"
#include "../PqaCore/Interface/PqaCommon.h"

namespace ProbQA {

template<typename taNumber> class CEUpdatePriorsSubtaskMul : public SRPlat::SRBaseSubtask {
  const TPqaId _iFirstTarget;
  const TPqaId _iLimTarget;

public: // methods
  CEUpdatePriorsSubtaskMul(CEUpdatePriorsTask<taNumber> *pTask, const TPqaId iFirstTarget, const TPqaId iLimTarget);
  virtual void Run() override final;
};

} // namespace ProbQA
