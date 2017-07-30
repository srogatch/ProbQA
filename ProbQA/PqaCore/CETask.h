// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/PqaCommon.h"
#include "../PqaCore/Interface/PqaErrorParams.h"

namespace ProbQA {

template<typename taNumber> class CpuEngine;

//TODO: to avoid excessive templating, consider making a base class for CpuEngine, containing the data that is not
//  specific to number type, i.e. not templated with taNumber . Then CETask could store a pointer to base non-templated
//  CpuEngine.
template<typename taNumber> class CETask : public SRPlat::SRBaseTask {
public: // types
  typedef SRPlat::SRSpinSync<16> TSync;

private: // variables
  AggregateErrorParams _aep; // guarded by _sync
  CpuEngine<taNumber> *_pCe;
  TSync _sync;
  const SRPlat::SRThreadPool::TThreadCount _nWorkers;

public: // variables
  explicit CETask(const SRPlat::SRThreadPool::TThreadCount nWorkers, CpuEngine<taNumber> *pCe)
    : _pCe(pCe), _nWorkers(nWorkers) { }

  virtual SRPlat::SRThreadPool& GetThreadPool() const override;

  void AddError(PqaError&& pe) {
    SRPlat::SRLock<TSync> sl(_sync);
    _aep.Add(std::forward<PqaError>(pe));
  }

  CpuEngine<taNumber>* GetEngine() const { return _pCe; }
  SRPlat::SRThreadPool::TThreadCount GetWorkerCount() const { return _nWorkers; }

  //NOTE: it's not thread-safe.
  PqaError TakeAggregateError(SRPlat::SRString &&message = SRPlat::SRString()) {
    if (_aep.Count() == 0) {
      return PqaError();
    }
    return PqaError(PqaErrorCode::Aggregate, _aep.Move(), std::forward<SRPlat::SRString>(message));
  }
};

} // namespace ProbQA
