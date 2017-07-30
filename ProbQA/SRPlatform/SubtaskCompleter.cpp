// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../SRPlatform/SubtaskCompleter.h"
#include "../SRPlatform/Interface/SRBaseSubtask.h"
#include "../SRPlatform/Interface/SRBaseTask.h"
#include "../SRPlatform/Interface/SRException.h"
#include "../SRPlatform/Interface/Exceptions/SRGenericException.h"
#include "../SRPlatform/Interface/Exceptions/SRStdException.h"
#include "../SRPlatform/Interface/SRThreadPool.h"
#include "../SRPlatform/Interface/SRLogStream.h"

namespace SRPlat {

#define STCLOG(pTaskVar, severityVar) \
  SRLogStream(ISRLogger::Severity::severityVar, pTaskVar->GetThreadPool().GetLogger())

SubtaskCompleter::~SubtaskCompleter() {
  if (_pSubtask != nullptr) {
    SRBaseTask *pTask = _pSubtask->GetTask();
    try {
      // Can't use subtask after the call below.
      pTask->FinalizeSubtask(_pSubtask);
    }
    catch (SRException& ex) {
      STCLOG(pTask, Error) << "SRException when finalizing a task: " << ex.ToString();
      pTask->HandleTaskFailure(std::move(ex));
    }
    catch (std::exception& ex) {
      STCLOG(pTask, Error) << "std::exception when finalizing a task: " << ex.what();
      pTask->HandleTaskFailure(SRStdException(ex));
    }
    catch (...) {
      std::exception_ptr ep = std::current_exception();
      STCLOG(pTask, Error) << "Unknown exception  when finalizing a task.";
      pTask->HandleTaskFailure(SRGenericException(ep));
    }
  }
}

} // namespace SRPlat
