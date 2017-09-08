// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CETrainTask.fwd.h"

namespace ProbQA {

template<typename taNumber> class CETrainSubtaskAdd : public SRBaseSubtask {
public: // types
  typedef CETrainTask<taNumber> TTask;

private: // variables
  SRPlat::SRThreadCount _iWorker;

public: // methods
  CETrainSubtaskAdd(TTask *pTask, const SRPlat::SRThreadCount iWorker)
    : SRBaseSubtask(pTask), _iWorker(iWorker) { }
  virtual void Run() override final;
};

} // namespace ProbQA
