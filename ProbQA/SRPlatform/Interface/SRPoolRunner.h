// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRStandardSubtask.h"
#include "../SRPlatform/Interface/SRThreadPool.h"

namespace SRPlat {

class SRPoolRunner {
public: // types
  template<typename taSubtask> class Keeper {
    friend class SRPoolRunner;

    taSubtask *const _pSubtasks;
    typename taSubtask::TTask *_pTask;
    SRThreadCount _nSubtasks = 0;

    //NOTE: this method doesn't set variables in a way to prevent another ReleaseInternal() operation.
    void ReleaseInternal() {
      if (_pTask) {
        _pTask->WaitComplete();
      }
      for (SRThreadCount i = 0; i < _nSubtasks; i++) {
        _pSubtasks[i].~taSubtask();
      }
    }

  public:
    explicit Keeper(void *pSubtasksMem, typename taSubtask::TTask& task)
      : _pSubtasks(SRCast::Ptr<taSubtask>(pSubtasksMem)), _pTask(&task) { }

    ~Keeper() {
      ReleaseInternal();
    }

    Keeper(const Keeper&) = delete;
    Keeper& operator=(const Keeper&) = delete;

    Keeper(Keeper&& source) : _pSubtasks(source._pSubtasks), _pTask(source._pTask), _nSubtasks(source._nSubtasks)
    {
      source._pTask = nullptr;
      source._nSubtasks = 0;
    }

    Keeper& operator=(Keeper&& source) {
      if (this != &source) {
        ReleaseInternal();
        _pSubtasks = source._pSubtasks;
        _pTask = source._pTask;
        _nSubtasks = source._nSubtasks;
        source._pTask = nullptr;
        source._nSubtasks = 0;
      }
      return *this;
    }

    taSubtask *GetSubtask(const SRThreadCount at) { return _pSubtasks + at; }
    SRThreadCount GetNSubtasks() const { return _nSubtasks; }
  };

  struct Split {
    // The first subtask starts at 0, the last subtask ends at the last bound. The other bounds are both the start of
    //   the next subtask and the limit of the previous subtask.
    size_t *const _pBounds;
    const SRThreadCount _nSubtasks;

    Split(size_t *const pBounds, const SRThreadCount nSubtasks) : _pBounds(pBounds), _nSubtasks(nSubtasks) { }
    void RecalcToStarts() {
      //TODO: vectorize, can't copy because regions overlap
      for (SRThreadCount i = _nSubtasks - 1; i >= 1; i--) {
        _pBounds[i] = _pBounds[i - 1];
      }
      _pBounds[0] = 0;
    }
  };

private: // variables
  SRThreadPool *_pTp;
  void *_pSubtasksMem;

public:
  // Returns the amount of memory required without padding. The allocating client must still pad the memory to SIMD size
  static size_t CalcSplitMemReq(const SRThreadCount nWorkers) {
    return nWorkers * sizeof(size_t);
  }

  explicit SRPoolRunner(SRThreadPool& tp, void *pSubtasksMem) : _pTp(&tp), _pSubtasksMem(pSubtasksMem) { }

  SRThreadPool& GetThreadPool() const { return *_pTp; }

  //TODO: vectorize, and align&pad each split memory
  static Split CalcSplit(void *pSplitMem, const size_t nItems, const SRThreadCount nWorkers) {
    size_t *pBounds = static_cast<size_t*>(pSplitMem);
    SRThreadCount nSubtasks = 0;

    size_t nextStart = 0;
    const lldiv_t perWorker = div((long long)nItems, (long long)nWorkers);
    while (nSubtasks < nWorkers && nextStart < nItems) {
      nextStart += perWorker.quot + (((long long)nSubtasks < perWorker.rem) ? 1 : 0);
      assert(nextStart <= nItems);
      pBounds[nSubtasks] = nextStart;
      nSubtasks++;
    }
    return {pBounds, nSubtasks};
  }

  //NOTE: use the static version to specify worker count explicitly. This version takes it from thread pool.
  Split CalcSplit(void *pSplitMem, const size_t nItems) {
    return CalcSplit(pSplitMem, nItems, _pTp->GetWorkerCount());
  }

  template<typename taSubtask, typename taCallback> inline Keeper<taSubtask> RunPreSplit(
    typename taSubtask::TTask& task, const Split& split, const taCallback &subtaskInit);

  template<typename taSubtask> inline Keeper<taSubtask> RunPreSplit(
    typename taSubtask::TTask& task, const Split& split)
  {
    return RunPreSplit<taSubtask>(task, split,
      [&](void *pStMem, const SRThreadCount iWorker, const int64_t iFirst, const int64_t iLimit) {
        taSubtask *pSt = new(pStMem) taSubtask(&task);
        pSt->SetStandardParams(iWorker, iFirst, iLimit);
      }
    );
  }

  template<typename taSubtask, typename taCallback> inline Keeper<taSubtask> SplitAndRunSubtasks(
    typename taSubtask::TTask& task, const size_t nItems, const SRThreadCount nWorkers, const taCallback &subtaskInit);

  template<typename taSubtask> inline Keeper<taSubtask> SplitAndRunSubtasks(typename taSubtask::TTask& task,
    const size_t nItems, const SRThreadCount nWorkers)
  {
    return SplitAndRunSubtasks<taSubtask>(task, nItems, nWorkers,
      [&](void *pStMem, const SRThreadCount iWorker, const int64_t iFirst, const int64_t iLimit) {
        taSubtask *pSt = new(pStMem) taSubtask(&task);
        pSt->SetStandardParams(iWorker, iFirst, iLimit);
      }
    );
  }

  template<typename taSubtask> inline Keeper<taSubtask> SplitAndRunSubtasks(typename taSubtask::TTask& task,
    const size_t nItems)
  {
    return SplitAndRunSubtasks<taSubtask>(task, nItems, _pTp->GetWorkerCount());
  }

  template<typename taSubtask, typename taCallback> inline Keeper<taSubtask> RunPerWorkerSubtasks(
    typename taSubtask::TTask& task, const SRThreadCount nWorkers, const taCallback &subtaskInit);

  template<typename taSubtask> inline Keeper<taSubtask> RunPerWorkerSubtasks(typename taSubtask::TTask& task,
    const SRThreadCount nWorkers)
  {
    return RunPerWorkerSubtasks<taSubtask>(task, nWorkers, [&](void *pStMem, const SRThreadCount iWorker) {
      taSubtask *pSt = new(pStMem) taSubtask(&task);
      pSt->SetStandardParams(iWorker, iWorker, iWorker + 1);
    });
  }
};

template<typename taSubtask, typename taCallback> inline SRPoolRunner::Keeper<taSubtask> SRPoolRunner::RunPreSplit(
  typename taSubtask::TTask& task, const Split& split, const taCallback &subtaskInit)
{
  task.Reset();
  Keeper<taSubtask> kp(_pSubtasksMem, task);
  size_t curStart = 0;
  while(kp._nSubtasks < split._nSubtasks) {
    const size_t nextStart = split._pBounds[kp._nSubtasks];
    subtaskInit(kp.GetSubtask(kp._nSubtasks), kp._nSubtasks, curStart, nextStart);
    // For finalization, it's important to increment subtask counter right after another subtask has been constructed.
    kp._nSubtasks++;
    curStart = nextStart;
  }
  _pTp->EnqueueAdjacent(kp._pSubtasks, kp._nSubtasks, task);

  kp._pTask = nullptr; // Don't call again SRBaseTask::WaitComplete() if it throws here.
  task.WaitComplete();
  return std::move(kp);
}

template<typename taSubtask, typename taCallback> inline
  SRPoolRunner::Keeper<taSubtask> SRPoolRunner::SplitAndRunSubtasks(
  typename taSubtask::TTask& task, const size_t nItems, const SRThreadCount nWorkers, const taCallback &subtaskInit)
{
  task.Reset();
  Keeper<taSubtask> kp(_pSubtasksMem, task);

  size_t nextStart = 0;
  const lldiv_t perWorker = div((long long)nItems, (long long)nWorkers);

  while (kp._nSubtasks < nWorkers && nextStart < nItems) {
    const size_t curStart = nextStart;
    nextStart += perWorker.quot + (((long long)kp._nSubtasks < perWorker.rem) ? 1 : 0);
    assert(nextStart <= nItems);
    subtaskInit(kp.GetSubtask(kp._nSubtasks), kp._nSubtasks, curStart, nextStart);
    // For finalization, it's important to increment subtask counter right after another subtask has been constructed.
    kp._nSubtasks++;
  }
  _pTp->EnqueueAdjacent(kp._pSubtasks, kp._nSubtasks, task);

  kp._pTask = nullptr; // Don't call again SRBaseTask::WaitComplete() if it throws here.
  task.WaitComplete();
  return std::move(kp);
}

template<typename taSubtask, typename taCallback> inline
  SRPoolRunner::Keeper<taSubtask> SRPoolRunner::RunPerWorkerSubtasks(
  typename taSubtask::TTask& task, const SRThreadCount nWorkers, const taCallback &subtaskInit)
{
  task.Reset();
  Keeper<taSubtask> kp(_pSubtasksMem, task);

  while (kp._nSubtasks < nWorkers) {
    subtaskInit(kp.GetSubtask(kp._nSubtasks), kp._nSubtasks);
    // For finalization, it's important to increment subtask counter right after another subtask has been constructed.
    kp._nSubtasks++;
  }
  _pTp->EnqueueAdjacent(kp._pSubtasks, kp._nSubtasks, task);

  kp._pTask = nullptr; // Don't call again SRBaseTask::WaitComplete() if it throws here.
  task.WaitComplete();
  return std::move(kp);
}

} // namespace SRPlat
