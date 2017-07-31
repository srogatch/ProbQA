// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

namespace ProbQA {

template<typename taNumber> class CETrainTask;

template<typename taNumber> class CETrainSubtaskAdd : public SRBaseSubtask {
public: // types
  typedef CETrainTask<taNumber> TTask;

private: // variables
  SRPlat::SRThreadPool::TThreadCount _iWorker;

public: // methods
  CETrainSubtaskAdd(TTask *pTask, const SRPlat::SRThreadPool::TThreadCount iWorker)
    : SRBaseSubtask(pTask), _iWorker(iWorker) { }
  virtual void Run() override final;
};

} // namespace ProbQA
