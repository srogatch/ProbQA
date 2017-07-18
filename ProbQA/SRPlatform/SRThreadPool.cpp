// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../SRPlatform/Interface/SRBaseSubtask.h"
#include "../SRPlatform/Interface/SRBaseTask.h"
#include "../SRPlatform/Interface/SRThreadPool.h"
#include "../SRPlatform/SubtaskCompleter.h"
#include "../SRPlatform/Interface/SRLock.h"
#include "../SRPlatform/Interface/SRException.h"
#include "../SRPlatform/Interface/SRDefaultLogger.h"
#include "../SRPlatform/Interface/SRLogStream.h"
#include "../SRPlatform/Interface/Exceptions/SRStdException.h"
#include "../SRPlatform/Interface/Exceptions/SRGenericException.h"
#include "../SRPlatform/Interface/SRUtils.h"

namespace SRPlat {

#define TPLOG(severityVar) SRLogStream(ISRLogger::Severity::severityVar, _pLogger.load(std::memory_order_acquire))

SRThreadPool::SRThreadPool(const size_t nThreads) : _shutdownRequested(0), _pLogger(SRDefaultLogger::Get()),
  _cbCritical(&DefaultCriticalCallback), _pCcbData(this)
{
  _workers.reserve(nThreads);
  for (size_t i = 0; i < nThreads; i++) {
    _workers.emplace_back(&SRThreadPool::WorkerEntry, this);
  }
}

void SRThreadPool::SetLogger(ISRLogger *pLogger) {
  if (pLogger == nullptr) {
    pLogger = SRDefaultLogger::Get();
  }
  _pLogger.store(pLogger, std::memory_order_relaxed);
}

void SRThreadPool::SetCriticalCallback(FCriticalCallback f, void *pData) {
  SRLock<SRCriticalSection> csl(_cs);
  if (f == nullptr) {
    _cbCritical = &DefaultCriticalCallback;
    _pCcbData = this;
  }
  else {
    _cbCritical = f;
    _pCcbData = pData;
  }
}

bool SRThreadPool::DefaultCriticalCallback(void *, SRException &&) {
  SRUtils::ExitProgram(SRExitCode::ThreadPoolCritical);
}

bool SRThreadPool::RunCriticalCallback(SRException &&ex) {
  FCriticalCallback f;
  void *p;
  {
    SRLock<SRCriticalSection> csl(_cs);
    f = _cbCritical;
    p = _pCcbData;
  }
  return f(p, std::forward<SRException>(ex));
}

#define DECIDE_CCB(exVar) if(RunCriticalCallback(exVar)) { continue; } else { break; }

void SRThreadPool::WorkerEntry() {
  for (;;) {
    SRBaseTask *pTask = nullptr;
    SubtaskCompleter stc;
    try {
      SRLock<SRCriticalSection> csl(_cs);
      while (_qu.size() == 0) {
        if (_shutdownRequested) {
          return;
        }
        _haveWork.Wait(_cs);
      }
      stc.Set(_qu.front());
      pTask = stc.Get()->GetTask();
      _qu.pop();
    }
    catch (SRException& ex) {
      TPLOG(Critical) << "Worker thread got an SRException in worker body: " << ex.ToString();
      DECIDE_CCB(std::move(ex));
    }
    catch (std::exception& ex) {
      TPLOG(Critical) << "Worker thread got an std::exception in worker body: " << ex.what();
      DECIDE_CCB(SRStdException(ex));
    }
    catch (...) {
      std::exception_ptr ep = std::current_exception();
      TPLOG(Critical) << "Worker thread got an unknown exception in worker body.";
      DECIDE_CCB(SRGenericException(ep));
    }

    try {
      stc.Get()->Run();
    }
    catch (SRException& ex) {
      TPLOG(Error) << "Worker thread got an SRException not handled in SRBaseTask::Run(): " << ex.ToString();
      pTask->HandleSubtaskFailure(std::move(ex), stc.Get());
    }
    catch (std::exception& ex) {
      TPLOG(Error) << "Worker thread got an std::exception not handled in SRBaseTask::Run(): " << ex.what();
      pTask->HandleSubtaskFailure(SRStdException(ex), stc.Get());
    }
    catch (...) {
      std::exception_ptr ep = std::current_exception();
      TPLOG(Error) << "Worker thread got an unknown exception not handled in SRBaseTask::Run().";
      pTask->HandleSubtaskFailure(SRGenericException(ep), stc.Get());
    }
  }
}

} // namespace SRPlat
