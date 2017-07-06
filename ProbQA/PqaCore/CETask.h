// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/PqaCommon.h"
#include "../PqaCore/Interface/PqaErrorParams.h"

namespace ProbQA {

template<typename taNumber> class CpuEngine;
template<typename taNumber> class CESubtask;

template<typename taNumber> class CETask {
public: // types
  typedef SRPlat::SRSpinSync<16> TSync;

private: // variables
  CpuEngine<taNumber> *_pCe;
  // Number of active subtasks
  std::atomic<TPqaId> _nToDo;
  TSync _sync;
  std::atomic<bool> _bCancel;
  std::unique_ptr<AggregateErrorParams> _pAep; // guarded by _sync

public: // variables
  SRPlat::SRConditionVariable _isComplete;

public: // variables
  explicit CETask(CpuEngine<taNumber> *pCe, const TPqaId nToDo = 0);
  // Returns the value after the increment.
  TPqaId IncToDo(const TPqaId by = 1) { return by + _nToDo.fetch_add(by, std::memory_order_relaxed); }
  TPqaId GetToDo() const { return _nToDo.load(std::memory_order_relaxed); }
  void Cancel() { _bCancel.store(true, std::memory_order_relaxed); }
  bool IsCancelled() const { return _bCancel.load(std::memory_order_relaxed); }
  void OnSubtaskComplete(CESubtask<taNumber> *pSubtask);

  void AddError(PqaError&& pe) {
    SRPlat::SRLock<TSync> sl(_sync);
    _pAep->Add(std::forward<PqaError>(pe));
  }
  //NOTE: it's not thread-safe, and it makes _pAep=nullptr if an error occured.
  PqaError TakeAggregateError(SRPlat::SRString &&message = SRPlat::SRString()) {
    if (_pAep->Count() == 0) {
      return PqaError();
    }
    return PqaError(PqaErrorCode::Aggregate, _pAep.release(), std::forward<SRPlat::SRString>(message));
  }

  void PrepareNextPhase() {
    if (_pAep == nullptr) {
      _pAep.reset(new AggregateErrorParams());
    }
    _bCancel.store(false, std::memory_order_relaxed);
  }
};

} // namespace ProbQA
