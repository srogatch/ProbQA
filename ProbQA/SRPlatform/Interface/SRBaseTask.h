// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRConditionVariable.h"

namespace SRPlat {

class SRBaseSubtask;
class SRThreadPool;

class SRBaseTask {
public: // types
  typedef int32_t TNSubtasks;

private: // variables
  std::atomic<TNSubtasks> _nToDo;
  std::atomic<TNSubtasks> _nFailedSubtasks;
  SRConditionVariable _isComplete;
  SRThreadPool *_pTp;

public: // methods
  virtual ~SRBaseTask() { }

  void FinalizeSubtask(SRBaseSubtask *pSubtask);
  // A hook for derived classes to e.g. release the subtask to a memory pool
  virtual void OnSubtaskComplete(SRBaseSubtask*) { };

  void HandlSubtaskError(SRBaseSubtask* pSubtask, const bool isFirstErr);
  virtual void OnSubtaskError(SRBaseSubtask*) { };
};

} // namespace SRPlat
