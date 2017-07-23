// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRConditionVariable.h"
#include "../SRPlatform/Interface/SRException.h"
#include "../SRPlatform/Interface/SRSpinSync.h"
#include "../SRPlatform/Interface/Exceptions/SRMultiException.h"

namespace SRPlat {

class SRBaseSubtask;
class SRThreadPool;

class SRPLATFORM_API SRBaseTask {
  friend class SRThreadPool;

public: // types
  typedef int32_t TNSubtasks;

private: // variables
  TNSubtasks _nToDo; // guarded by the critical section of the thread pool
  // It can be a little more than the number of subtasks, if failures happen in the task code too.
  std::atomic<TNSubtasks> _nFailures;
  SRConditionVariable _isComplete;
  //// Cache-insensitive data
  SRThreadPool *_pTp;

public: // methods
  explicit SRBaseTask(SRThreadPool *pTp);
  virtual ~SRBaseTask() { }

  void FinalizeSubtask(SRBaseSubtask *pSubtask);
  // A hook for derived classes to e.g. release the subtask to a memory pool
  virtual void OnSubtaskComplete(SRBaseSubtask*) { };

  void HandleSubtaskFailure(SRException &&ex, SRBaseSubtask* pSubtask);
  virtual void OnSubtaskFailure(SRException &&, SRBaseSubtask*) { };

  void HandleTaskFailure(SRException &&ex);
  virtual void OnTaskFailure(SRException &&) { }

  void WaitComplete();

  SRThreadPool* GetThreadPool() const {
    return _pTp;
  }
};

} // namespace SRPlat
