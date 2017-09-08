// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CETask.fwd.h"
#include "../PqaCore/CEBaseTask.h"
#include "../PqaCore/Interface/PqaCommon.h"
#include "../PqaCore/Interface/PqaErrorParams.h"

namespace ProbQA {

class CETask : public CEBaseTask {
public: // types
  typedef SRPlat::SRSpinSync<16> TSync;

private: // variables
  AggregateErrorParams _aep; // guarded by _sync
  TSync _sync;
  const SRPlat::SRThreadCount _nWorkers;

public: // variables
  explicit CETask(BaseCpuEngine &bce, const SRPlat::SRThreadCount nWorkers);
  // This method returns the worker count allocated for the current task. The thread pool may have a different worker
  //   count.
  SRPlat::SRThreadCount GetWorkerCount() const;
  void AddError(PqaError&& pe);
  //NOTE: it's not thread-safe.
  PqaError TakeAggregateError(SRPlat::SRString &&message = SRPlat::SRString());
};

} // namespace ProbQA
