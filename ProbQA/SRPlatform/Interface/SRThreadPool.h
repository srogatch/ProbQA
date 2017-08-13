// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRCriticalSection.h"
#include "../SRPlatform/Interface/SRConditionVariable.h"
#include "../SRPlatform/Interface/ISRLogCustomizable.h"
#include "../SRPlatform/Interface/SRQueue.h"

namespace SRPlat {

class SRBaseSubtask;

class SRPLATFORM_API SRThreadPool : public ISRLogCustomizable {
  struct RareData; // cache-insensitive piece of thread pool data

public: // types
  // Returns |true| if worker thread should continue, or |false| if it should exit.
  typedef bool (*FCriticalCallback)(void *, SRException&&);
  typedef uint32_t TThreadCount;

private: // variables
  SRQueue<SRBaseSubtask*> _qu;
  SRCriticalSection _cs;
  SRConditionVariable _haveWork;
  // It has to be const to allow accessing without locks by the clients.
  const TThreadCount _nWorkers;
  uint8_t _shutdownRequested : 1;
  RareData *_pRd;

private: // methods
  void WorkerEntry();
  static bool DefaultCriticalCallback(void *pData, SRException &&ex);
  bool RunCriticalCallback(SRException &&ex);

public:
  explicit SRThreadPool(const TThreadCount nThreads = std::thread::hardware_concurrency());
  virtual ~SRThreadPool() override final;

  virtual ISRLogger* GetLogger() const override final;
  virtual void SetLogger(ISRLogger *pLogger) override final;

  // If f==nullptr , the function sets the default callback with |this| as data.
  void SetCriticalCallback(FCriticalCallback f, void *pData = nullptr);

  TThreadCount GetWorkerCount() const { return _nWorkers; }

  void Enqueue(SRBaseSubtask *pSt);
  // The subtasks can belong to different tasks.
  void __vectorcall Enqueue(std::initializer_list<SRBaseSubtask*> subtasks);
  // The subtasks must belong to the same task passed as a parameter.
  void __vectorcall Enqueue(std::initializer_list<SRBaseSubtask*> subtasks, SRBaseTask &task);

  // Request shutdown. This emthod doesn't wait for all threads to exit: only destructor does.
  void RequestShutdown();

  SRCriticalSection& GetCS() { return _cs; }
};

} // namespace SRPlat
