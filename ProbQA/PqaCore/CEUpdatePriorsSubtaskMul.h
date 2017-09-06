// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CEUpdatePriorsTask.fwd.h"
#include "../PqaCore/Interface/PqaCommon.h"

namespace ProbQA {

template<typename taNumber> class CEUpdatePriorsSubtaskMul : public SRPlat::SRBaseSubtask {
  const TPqaId _iFirstVT; // first vector of targets
  const TPqaId _iLimVT; // limit vector of targets

private: // methods
  template<bool taCache> void RunInternal(const CEUpdatePriorsTask<taNumber>& task) const;

public: // methods
  CEUpdatePriorsSubtaskMul(CEUpdatePriorsTask<taNumber> *pTask, const TPqaId iFirstVT, const TPqaId iLimVT);
  virtual void Run() override final;
};

} // namespace ProbQA
