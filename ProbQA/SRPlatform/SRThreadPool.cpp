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

namespace SRPlat {

#define TPLOG(severityVar) SRLogStream(ISRLogger::Severity::severityVar, _pLogger.load(std::memory_order_acquire))

SRThreadPool::SRThreadPool(const size_t nThreads) : _shutdownRequested(0), _pLogger(SRDefaultLogger::Get()) {
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

void SRThreadPool::WorkerEntry() {
  for (;;) {
    SRBaseTask *pTask = nullptr;
    try {
      SubtaskCompleter stc;
      {
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
    catch (SRException& ex) {
      TPLOG(Error) << "Worker thread got an SRException in worker body: " << ex.ToString();
      pTask->HandleTaskFailure(std::move(ex));
    }
    catch (std::exception& ex) {
      TPLOG(Error) << "Worker thread got an std::exception in worker body: " << ex.what();
      pTask->HandleTaskFailure(SRStdException(ex));
    }
    catch (...) {
      std::exception_ptr ep = std::current_exception();
      TPLOG(Error) << "Worker thread got an unknown exception in worker body.";
      pTask->HandleTaskFailure(SRGenericException(ep));
    }
  }
}

} // namespace SRPlat
