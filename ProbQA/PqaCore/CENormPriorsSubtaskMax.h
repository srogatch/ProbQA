// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CENormPriorsTask.fwd.h"

namespace ProbQA {

template<typename taNumber> class CENormPriorsSubtaskMax : public SRPlat::SRStandardSubtask {
public: // types
  typedef CENormPriorsTask<taNumber> TTask;

public: // variables
  int64_t _maxExp; // Result

public: // methods
  explicit CENormPriorsSubtaskMax(TTask *pTask);
  virtual void Run() override final;
};

} // namespace ProbQA
