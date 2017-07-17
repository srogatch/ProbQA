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
      auto mb = SRMessageBuilder("A task got a negative number of remaining subtasks to do: ")(nOld - 1);
      _pTp->GetLogger()->Log(ISRLogger::Severity::Critical, mb.GetUnownedSRString());
      HandleTaskFailure(SRException(mb.GetOwnedSRString()));
    }
    _isComplete.WakeAll();
  }
  OnSubtaskComplete(pSubtask);
}

void SRBaseTask::HandleSubtaskFailure(SRException &&ex, SRBaseSubtask* pSubtask) {
  _nFailures.fetch_add(1, std::memory_order_relaxed);
  OnSubtaskFailure(std::forward<SRException>(ex), pSubtask);
}

void SRBaseTask::HandleTaskFailure(SRException &&ex) {
  _nFailures.fetch_add(1, std::memory_order_relaxed);
  OnTaskFailure(std::forward<SRException>(ex));
}

} // namespace SRPlat
