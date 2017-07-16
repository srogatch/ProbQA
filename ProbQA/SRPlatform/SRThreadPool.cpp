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
    SubtaskCompleter stc;
    try {
      {
        SRLock<SRCriticalSection> csl(_cs);
        while (_qu.size() == 0) {
          if (_shutdownRequested) {
            return;
          }
          _haveWork.Wait(_cs);
        }
        stc.Set(_qu.front());
        _qu.pop();
      }
      stc.Get()->Run();
    }
    catch (SRException& ex) {
      TPLOG(Error) << "Worker thread got an SRException not handled in the call subtree: " << ex.ToString();
      //stc.Get()->
      //ceStc.Get()->_pTask->AddError(std::move(err));
    }
    catch (std::exception& ex) {
      //PqaError err;
      //err.SetFromException(std::move(ex));
      //CELOG(Critical) << "Worker thread got an std::exception not handled at lower levels: " << err.ToString(true);
      //ceStc.Get()->_pTask->AddError(std::move(err));
    }
    catch (...) {
      std::exception_ptr ep = std::current_exception();
      //ep.
    }
  }
}

} // namespace SRPlat
