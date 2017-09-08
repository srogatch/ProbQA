// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRConditionVariable.h"
#include "../SRPlatform/Interface/SRException.h"
#include "../SRPlatform/Interface/SRSpinSync.h"
#include "../SRPlatform/Interface/Exceptions/SRMultiException.h"
#include "../SRPlatform/Interface/SRBasicTypes.h"

namespace SRPlat {

class SRBaseSubtask;
class SRThreadPool;

class SRPLATFORM_API SRBaseTask {
  friend class SRThreadPool;

private: // variables
  SRSubtaskCount _nToDo; // guarded by the critical section of the thread pool
  // It can be a little more than the number of subtasks, if failures happen in the task code too.
  std::atomic<SRSubtaskCount> _nFailures;
  SRConditionVariable _isComplete;

public: // methods
  //NOTE: the destructor doesn't call WaitComplete() for performance reasons (save extra enter/leave CS) and because
  //  the derived object is already destructed, as well as may some satellite objects handling On*() events. Client
  //  code must ensure WaitComplete() is called before destructing a task that has subtasks running.
  virtual ~SRBaseTask() { }

  void FinalizeSubtask(SRBaseSubtask *pSubtask);
  // A hook for derived classes to e.g. release the subtask to a memory pool
  virtual void OnSubtaskComplete(SRBaseSubtask*) { };

  void HandleSubtaskFailure(SRException &&ex, SRBaseSubtask* pSubtask);
  virtual void OnSubtaskFailure(SRException &&, SRBaseSubtask*) { };

  void HandleTaskFailure(SRException &&ex);
  virtual void OnTaskFailure(SRException &&) { }

  void WaitComplete();

  virtual SRThreadPool& GetThreadPool() const = 0;
};

} // namespace SRPlat
