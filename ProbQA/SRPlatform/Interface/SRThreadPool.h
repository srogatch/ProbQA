// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRCriticalSection.h"
#include "../SRPlatform/Interface/SRConditionVariable.h"
#include "../SRPlatform/Interface/ISRLogCustomizable.h"
#include "../SRPlatform/Interface/SRQueue.h"
#include "../SRPlatform/Interface/SRBasicTypes.h"
#include "../SRPlatform/Interface/SRLock.h"

namespace SRPlat {

class SRBaseSubtask;
class SRBaseTask;
template class SRPLATFORM_API SRQueue<SRBaseSubtask*>;

class SRPLATFORM_API SRThreadPool : public ISRLogCustomizable {
  struct RareData; // cache-insensitive piece of thread pool data

public: // types
  // Returns |true| if worker thread should continue, or |false| if it should exit.
  typedef bool (*FCriticalCallback)(void *, SRException&&);

private: // variables
  SRQueue<SRBaseSubtask*> _qu;
  SRCriticalSection _cs;
  SRConditionVariable _haveWork;
  // It has to be const to allow accessing without locks by the clients.
  const SRThreadCount _nWorkers;
  uint8_t _shutdownRequested : 1;
  RareData *_pRd;

private: // methods
  void WorkerEntry();
  static bool DefaultCriticalCallback(void *pData, SRException &&ex);
  bool RunCriticalCallback(SRException &&ex);

public:
  explicit SRThreadPool(const SRThreadCount nThreads = std::thread::hardware_concurrency());
  virtual ~SRThreadPool() override final;

  virtual ISRLogger* GetLogger() const override final;
  virtual void SetLogger(ISRLogger *pLogger) override final;

  // If f==nullptr , the function sets the default callback with |this| as data.
  void SetCriticalCallback(FCriticalCallback f, void *pData = nullptr);

  SRThreadCount GetWorkerCount() const { return _nWorkers; }

  void Enqueue(SRBaseSubtask *pSt);
  // The subtasks can belong to different tasks.
  void __vectorcall Enqueue(std::initializer_list<SRBaseSubtask*> subtasks);
  // The subtasks must belong to the same task passed as a parameter.
  void __vectorcall Enqueue(std::initializer_list<SRBaseSubtask*> subtasks, SRBaseTask &task);

  // Subtasks must be adjacent in memory to one another, like: taSubtask sts[16];
  // Subtasks must belong to the same task.
  template<typename taSubtask> inline void EnqueueAdjacent(taSubtask *pFirst, const SRSubtaskCount nSubtasks,
    SRBaseTask &task);

  // Request shutdown. This method doesn't wait for all threads to exit: only destructor does.
  void RequestShutdown();

  SRCriticalSection& GetCS() { return _cs; }
};

template<typename taSubtask> inline void SRThreadPool::EnqueueAdjacent(taSubtask *pFirst, const SRSubtaskCount nSubtasks,
  SRBaseTask &task)
{
  {
    SRLock<SRCriticalSection> csl(_cs);
    if (_shutdownRequested) {
      throw SRException(SRString::MakeUnowned("An attempt to push multiple subtasks to a shut(ting) down thread pool."));
    }
    for (size_t i = 0; i < nSubtasks; i++) {
      _qu.Push(pFirst + i);
    }
    task._nToDo += nSubtasks;
  }
  _haveWork.WakeAll();
}

} // namespace SRPlat
