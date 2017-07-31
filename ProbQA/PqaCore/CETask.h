// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CETask.decl.h"
#include "../PqaCore/BaseCpuEngine.h"

namespace ProbQA {

inline CETask::CETask(const SRPlat::SRThreadPool::TThreadCount nWorkers, BaseCpuEngine *pCe)
  : _pCe(pCe), _nWorkers(nWorkers) { }

inline SRPlat::SRThreadPool& CETask::GetThreadPool() const {
  return _pCe->GetWorkers();
}

inline void CETask::AddError(PqaError&& pe) {
  SRPlat::SRLock<TSync> sl(_sync);
  _aep.Add(std::forward<PqaError>(pe));
}

inline BaseCpuEngine* CETask::GetEngine() const { return _pCe; }

inline SRPlat::SRThreadPool::TThreadCount CETask::GetWorkerCount() const { return _nWorkers; }

inline PqaError CETask::TakeAggregateError(SRPlat::SRString &&message) {
  if (_aep.Count() == 0) {
    return PqaError();
  }
  return PqaError(PqaErrorCode::Aggregate, _aep.Move(), std::forward<SRPlat::SRString>(message));
}

} // namespace ProbQA
