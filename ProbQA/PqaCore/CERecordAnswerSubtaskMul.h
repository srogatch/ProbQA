// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CERecordAnswerTask.fwd.h"
#include "../PqaCore/Interface/PqaCommon.h"

namespace ProbQA {

template<typename taNumber> class CERecordAnswerSubtaskMul : public SRPlat::SRStandardSubtask {
public: // types
  typedef CERecordAnswerTask<taNumber> TTask;

public: // variables
  taNumber _sumPriors;

public: // methods
  using SRPlat::SRStandardSubtask::SRStandardSubtask;
  virtual void Run() override final;
};

} // namespace ProbQA
