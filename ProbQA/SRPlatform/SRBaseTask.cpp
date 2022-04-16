// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "../SRPlatform/Interface/SRBaseTask.h"
#include "../SRPlatform/Interface/SRThreadPool.h"
#include "../SRPlatform/Interface/SRLogStream.h"
#include "../SRPlatform/Interface/SRLock.h"

namespace SRPlat {

#define BTLOG(severityVar) SRLogStream(ISRLogger::Severity::severityVar, GetThreadPool().GetLogger())

void SRBaseTask::FinalizeSubtask(SRBaseSubtask *pSubtask) {
  OnSubtaskComplete(pSubtask);
  SRSubtaskCount nNew;
  {
    SRLock<SRCriticalSection> csl(GetThreadPool().GetCS());
    nNew = --_nToDo;

    if (nNew <= 0) {
      if (nNew < 0) {
        csl.EarlyRelease();
        auto mb = SRMessageBuilder("A task got a negative number of remaining subtasks to do: ")(nNew);
        GetThreadPool().GetLogger()->Log(ISRLogger::Severity::Critical, mb.GetUnownedSRString());
        HandleTaskFailure(SRException(mb.GetOwnedSRString()));
        return;
      }
      //This must be executed while locked, otherwise the condition variable may be destructed concurrently.
      _isComplete.WakeAll();
    }
  }
}

void SRBaseTask::HandleSubtaskFailure(SRException &&ex, SRBaseSubtask* pSubtask) {
  _nFailures.fetch_add(1, std::memory_order_relaxed);
  OnSubtaskFailure(std::forward<SRException>(ex), pSubtask);
}

void SRBaseTask::HandleTaskFailure(SRException &&ex) {
  _nFailures.fetch_add(1, std::memory_order_relaxed);
  OnTaskFailure(std::forward<SRException>(ex));
}

void SRBaseTask::WaitComplete() {
  SRCriticalSection &cs = GetThreadPool().GetCS();
  SRLock<SRCriticalSection> csl(cs);
  while (_nToDo > 0) {
    _isComplete.Wait(cs);
  }
}

void SRBaseTask::Reset() {
  _nToDo = 0;
  _nFailures.store(0, std::memory_order_release);
}

} // namespace SRPlat
