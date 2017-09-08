// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRStandardSubtask.h"
#include "../SRPlatform/Interface/SRThreadPool.h"

namespace SRPlat {

template<typename taSubtask> class SRPoolRunner {
  SRThreadPool *_pTp;
  void *_pSubtasksMem;

public:
  explicit SRPoolRunner(SRThreadPool& tp, void *pSubtasksMem) : _pTp(&tp), _pSubtasksMem(pSubtasksMem) { }

  template<typename taCallback> inline void SplitAndRunSubtasks(typename taSubtask::TTask& task, const size_t nItems,
    const SRThreadCount nWorkers, const taCallback &subtaskInit);

  inline void SplitAndRunSubtasks(typename taSubtask::TTask& task, const size_t nItems,
    const SRThreadCount nWorkers)
  {
    return SplitAndRunSubtasks(task, nItems, nWorkers,
      [](void *pStMem, SRThreadCount iWorker, int64_t iFirst, int64_t iLimit) {
        taSubtask *pSt = new(pStMem) taSubtask(&task);
        pSt->SetStandardParams(iWorker, iFirst, iLimit);
      }
    );
  }

  inline void SplitAndRunSubtasks(typename taSubtask::TTask& task, const size_t nItems) {
    return SplitAndRunSubtasks(task, nItems, _pTp->GetWorkerCount());
  }
};

template<typename taSubtask> template<typename taCallback> inline void SRPoolRunner<taSubtask>::SplitAndRunSubtasks(
  typename taSubtask::TTask& task, const size_t nItems, const SRThreadCount nWorkers, const taCallback &subtaskInit)
{
  taSubtask *const pSubtasks = reinterpret_cast<taSubtask*>(_pSubtasksMem);
  SRThreadCount nSubtasks = 0;
  bool bWorkersFinished = false;

  auto&& subtasksFinally = SRMakeFinally([&] {
    if (!bWorkersFinished) {
      task.WaitComplete();
    }
    for (size_t i = 0; i < nSubtasks; i++) {
      pSubtasks[i].~taSubtask();
    }
  }); (void)subtasksFinally;

  size_t nextStart = 0;
  const lldiv_t perWorker = div((long long)nItems, (long long)nWorkers);

  while (nSubtasks < nWorkers && nextStart < nItems) {
    const size_t curStart = nextStart;
    nextStart += perWorker.quot + (((long long)nSubtasks < perWorker.rem) ? 1 : 0);
    assert(nextStart <= nItems);
    subtaskInit(pSubtasks + nSubtasks, nSubtasks, curStart, nextStart);
    // For finalization, it's important to increment subtask counter right after another subtask has been
    //   constructed.
    nSubtasks++;
  }
  _pTp->EnqueueAdjacent(pSubtasks, nSubtasks, task);

  bWorkersFinished = true; // Don't call again SRBaseTask::WaitComplete() if it throws here.
  task.WaitComplete();
}

} // namespace SRPlat

