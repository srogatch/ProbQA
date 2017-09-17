// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CEUpdatePriorsTask.fwd.h"
#include "../PqaCore/Interface/PqaCommon.h"

namespace ProbQA {

template<typename taNumber> class CEUpdatePriorsSubtaskMul : public SRPlat::SRStandardSubtask {
public: // types
  typedef CEUpdatePriorsTask<taNumber> TTask;

private: // methods
  template<bool taCache> void RunInternal(const TTask& task) const;

public: // methods
  explicit CEUpdatePriorsSubtaskMul(TTask *pTask);
  virtual void Run() override final;
};

} // namespace ProbQA
