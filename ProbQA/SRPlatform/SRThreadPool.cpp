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
#include "../SRPlatform/Interface/SRLogMacros.h"

namespace SRPlat {

#pragma warning( push )
// warning C4200: nonstandard extension used: zero-sized array in struct/union
// note: This member will be ignored by a defaulted constructor or copy/move assignment operator
#pragma warning( disable : 4200 )
struct SRThreadPool::RareData {
  std::atomic<ISRLogger*> _pLogger;
  FCriticalCallback _cbCritical;
  void *_pCcbData;
  HANDLE _workers[0];

  static DWORD WINAPI PlatformEntry(LPVOID lpParameter) {
    static_cast<SRThreadPool*>(lpParameter)->WorkerEntry();
    return 0;
  }
};
#pragma warning( pop )

#define TPLOG(severityVar) SRLogStream(ISRLogger::Severity::severityVar, GetLogger())

SRThreadPool::SRThreadPool(const SRThreadCount nThreads, const size_t stackSize) : _qu(SRMath::CeilLog2(nThreads)),
  _nWorkers(nThreads), _shutdownRequested(0), _stackSize(stackSize)
{
  _pRd = SRCast::Ptr<RareData>(malloc(sizeof(RareData) + sizeof(HANDLE) * _nWorkers));
  _pRd->_pLogger = SRDefaultLogger::Get();
  SetCriticalCallback(nullptr);
  LaunchThreads();
}

SRThreadPool::~SRThreadPool() {
  RequestShutdown();
  StopThreads();
  free(_pRd);
}

void SRThreadPool::LaunchThreads() {
  assert(!_shutdownRequested);
  for (size_t i = 0; i < _nWorkers; i++) {
    // Use WinAPI threads in order to be able to set the stack size
    HANDLE hThread = CreateThread(nullptr, _stackSize + _cReserveStackSize, &RareData::PlatformEntry, this, 0, nullptr);
    if (hThread == nullptr) {
      SR_LOG_WINFAIL_GLE(Critical, GetLogger());
      SRUtils::ExitProgram(SRExitCode::ThreadPoolCritical);
    }
    _pRd->_workers[i] = hThread;
  }
}

void SRThreadPool::StopThreads() {
  for (size_t i = 0; i < _nWorkers; i += MAXIMUM_WAIT_OBJECTS) {
    const uint32_t nToWait = std::min<uint32_t>(MAXIMUM_WAIT_OBJECTS, uint32_t(_nWorkers - i));
    const uint32_t waitRes = WaitForMultipleObjects(nToWait, _pRd->_workers + i, true, INFINITE);
    if (waitRes < WAIT_OBJECT_0 || waitRes >= WAIT_OBJECT_0 + nToWait) {
      TPLOG(Error) << SR_FILE_LINE "Got waitRes=" << waitRes << ", GetLastError()=" << GetLastError();
    }
    for (uint32_t j = 0; j < nToWait; j++) {
      if (!CloseHandle(_pRd->_workers[i + j])) {
        SR_LOG_WINFAIL_GLE(Error, GetLogger());
      }
    }
  }
}

void SRThreadPool::ChangeStackSize(const size_t stackSize) {
  RequestShutdown();
  StopThreads();
  _stackSize = stackSize;
  _shutdownRequested = 0;
  LaunchThreads();
}

void SRThreadPool::SetLogger(ISRLogger *pLogger) {
  if (pLogger == nullptr) {
    pLogger = SRDefaultLogger::Get();
  }
  _pRd->_pLogger.store(pLogger, std::memory_order_relaxed);
}

void SRThreadPool::SetCriticalCallback(FCriticalCallback f, void *pData) {
  SRLock<SRCriticalSection> csl(_cs);
  if (f == nullptr) {
    _pRd->_cbCritical = &DefaultCriticalCallback;
    _pRd->_pCcbData = this;
  }
  else {
    _pRd->_cbCritical = f;
    _pRd->_pCcbData = pData;
  }
}

ISRLogger* SRThreadPool::GetLogger() const {
  return _pRd->_pLogger.load(std::memory_order_relaxed);
}

bool SRThreadPool::DefaultCriticalCallback(void *, SRException &&) {
  SRUtils::ExitProgram(SRExitCode::ThreadPoolCritical);
  //return false; // just to please the compiler
}

bool SRThreadPool::RunCriticalCallback(SRException &&ex) {
  FCriticalCallback f;
  void *p;
  {
    SRLock<SRCriticalSection> csl(_cs);
    f = _pRd->_cbCritical;
    p = _pRd->_pCcbData;
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
      while (_qu.Size() == 0) {
        if (_shutdownRequested) {
          return;
        }
        _haveWork.Wait(_cs);
      }
      stc.Set(_qu.PopGet());
      pTask = stc.Get()->GetTask();
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

void SRThreadPool::Enqueue(SRBaseSubtask *pSt) {
  {
    SRLock<SRCriticalSection> csl(_cs);
    if (_shutdownRequested) {
      throw SRException(SRString::MakeUnowned("An attempt to push a subtask to a shut(ting) down thread pool."));
    }
    _qu.Push(pSt);
    pSt->GetTask()->_nToDo++;
  }
  _haveWork.WakeOne();
}

void __vectorcall SRThreadPool::Enqueue(std::initializer_list<SRBaseSubtask*> subtasks) {
  {
    SRLock<SRCriticalSection> csl(_cs);
    if (_shutdownRequested) {
      throw SRException(SRString::MakeUnowned("An attempt to push multiple subtasks to a shut(ting) down thread pool."));
    }
    _qu.Push(subtasks.begin(), subtasks.size());
    for (SRBaseSubtask *pSt : subtasks) {
      pSt->GetTask()->_nToDo++; // the subtasks may belong to different tasks
    }
  }
  _haveWork.WakeAll();
}

void __vectorcall SRThreadPool::Enqueue(std::initializer_list<SRBaseSubtask*> subtasks, SRBaseTask &task) {
  {
    SRLock<SRCriticalSection> csl(_cs);
    if (_shutdownRequested) {
      throw SRException(SRString::MakeUnowned("An attempt to push multiple subtasks to a shut(ting) down thread pool."));
    }
    _qu.Push(subtasks.begin(), subtasks.size());
    task._nToDo += static_cast<SRSubtaskCount>(subtasks.size());
  }
  _haveWork.WakeAll();
}

void SRThreadPool::RequestShutdown() {
  {
    SRLock<SRCriticalSection> csl(_cs);
    _shutdownRequested = 1;
  }
  _haveWork.WakeAll();
}

} // namespace SRPlat
