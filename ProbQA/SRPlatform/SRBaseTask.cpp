// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../SRPlatform/Interface/SRBaseTask.h"
#include "../SRPlatform/Interface/SRThreadPool.h"
#include "../SRPlatform/Interface/SRLogStream.h"

namespace SRPlat {

#define BTLOG(severityVar) SRLogStream(ISRLogger::Severity::severityVar, _pTp->GetLogger())

void SRBaseTask::FinalizeSubtask(SRBaseSubtask *pSubtask) {
  TNSubtasks nOld = _nToDo.fetch_sub(1, std::memory_order_release);
  if (nOld <= 1) {
    if (nOld <= 0) {
      BTLOG(Critical) << "A task got a negative number of remaining subtasks to do: " << (nOld - 1);
    }
    _isComplete.WakeAll();
  }
  FinalizeSubtask(pSubtask);
}

void SRBaseTask::HandlSubtaskError(SRBaseSubtask* pSubtask, const bool isFirstErr) {
  if (isFirstErr) {
    _nFailedSubtasks.fetch_add(1, std::memory_order_release);
  }
  OnSubtaskError(pSubtask);
}

} // namespace SRPlat
