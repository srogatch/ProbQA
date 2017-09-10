// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CENormPriorsTask.fwd.h"
#include "../PqaCore/Interface/PqaCommon.h"

namespace ProbQA {

template<typename taNumber> class CENormPriorsSubtaskMax : public SRPlat::SRStandardSubtask {
public: // types
  typedef CENormPriorsTask<taNumber> TTask;

public: // methods
  explicit CENormPriorsSubtaskMax(CENormPriorsTask<taNumber> *pTask);
  virtual void Run() override final;
};

} // namespace ProbQA
