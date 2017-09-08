// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CETrainTask.fwd.h"

namespace ProbQA {

template<typename taNumber> class CETrainSubtaskAdd : public SRStandardSubtask {
public: // types
  typedef CETrainTask<taNumber> TTask;

public: // methods
  explicit CETrainSubtaskAdd(TTask *pTask) : SRStandardSubtask(pTask) { }
  virtual void Run() override final;
};

} // namespace ProbQA
