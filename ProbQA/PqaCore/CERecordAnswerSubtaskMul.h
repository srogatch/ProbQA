// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CERecordAnswerTask.fwd.h"
#include "../PqaCore/Interface/PqaCommon.h"

namespace ProbQA {

template<typename taNumber> class CERecordAnswerSubtaskMul : public SRPlat::SRStandardSubtask {
public:
  typedef CERecordAnswerTask<taNumber> TTask;

public:
  explicit CERecordAnswerSubtaskMul(TTask *pTask);
  virtual void Run() override final;
};

} // namespace ProbQA
