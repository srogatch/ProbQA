// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/BaseCpuEngine.decl.h"
#include "../PqaCore/CETask.h"

namespace ProbQA {

template<typename taSubtask, typename taCallback> inline PqaError BaseCpuEngine::SplitAndRunSubtasks(
  CETask &task, const size_t nItems, void *pSubtaskMem, const taCallback &subtaskPlNew)
{
  taSubtask *const pSubtasks = reinterpret_cast<taSubtask*>(pSubtaskMem);
  const SRPlat::SRThreadCount nWorkers = task.GetWorkerCount();
  SRPlat::SRThreadCount nSubtasks = 0;
  size_t nextStart = 0;
  const lldiv_t perWorker = div((long long)nItems, (long long)nWorkers);
  bool bWorkersFinished = false;

  auto&& subtasksFinally = SRPlat::SRMakeFinally([&] {
    if (!bWorkersFinished) {
      task.WaitComplete();
    }
    for (size_t i = 0; i < nSubtasks; i++) {
      pSubtasks[i].~taSubtask();
    }
  }); (void)subtasksFinally;

  while (nSubtasks < nWorkers && nextStart < nItems) {
    size_t curStart = nextStart;
    nextStart += perWorker.quot;
    if ((long long)nSubtasks < perWorker.rem) {
      nextStart++;
    }
    assert(nextStart <= nItems);
    subtaskPlNew(pSubtasks + nSubtasks, curStart, nextStart);
    // For finalization, it's important to increment subtask counter right after another subtask has been
    //   constructed.
    nSubtasks++;
    _tpWorkers.Enqueue(pSubtasks + nSubtasks - 1);
  }

  bWorkersFinished = true; // Don't call again SRBaseTask::WaitComplete() if it throws here.
  task.WaitComplete();

  return task.TakeAggregateError(SRString::MakeUnowned("Failed " SR_FILE_LINE));
}

template<typename taSubtask, typename taCallback> void BaseCpuEngine::SplitAndRunSubtasksSlim(CEBaseTask &task,
  const size_t nItems, void *pSubtaskMem, const taCallback &subtaskPlNew)
{
  taSubtask *const pSubtasks = reinterpret_cast<taSubtask*>(pSubtaskMem);
  const SRPlat::SRThreadCount nWorkers = _tpWorkers.GetWorkerCount();
  SRPlat::SRThreadCount nSubtasks = 0;
  size_t nextStart = 0;
  const lldiv_t perWorker = div((long long)nItems, (long long)nWorkers);
  bool bWorkersFinished = false;

  auto&& subtasksFinally = SRPlat::SRMakeFinally([&] {
    if (!bWorkersFinished) {
      task.WaitComplete();
    }
    for (size_t i = 0; i < nSubtasks; i++) {
      pSubtasks[i].~taSubtask();
    }
  }); (void)subtasksFinally;

  while (nSubtasks < nWorkers && nextStart < nItems) {
    size_t curStart = nextStart;
    nextStart += perWorker.quot;
    if ((long long)nSubtasks < perWorker.rem) {
      nextStart++;
    }
    assert(nextStart <= nItems);
    subtaskPlNew(pSubtasks + nSubtasks, curStart, nextStart);
    // For finalization, it's important to increment subtask counter right after another subtask has been
    //   constructed.
    nSubtasks++;
    _tpWorkers.Enqueue(pSubtasks + nSubtasks - 1);
  }

  bWorkersFinished = true; // Don't call again SRBaseTask::WaitComplete() if it throws here.
  task.WaitComplete();
}

template<typename taSubtask> inline
PqaError BaseCpuEngine::RunWorkerOnlySubtasks(typename taSubtask::TTask &task, void *pSubtaskMem) {
  taSubtask *const pSubtasks = reinterpret_cast<taSubtask*>(pSubtaskMem);
  const SRThreadCount nWorkers = task.GetWorkerCount();
  SRThreadCount nSubtasks = 0;
  bool bWorkersFinished = false;

  auto&& subtasksFinally = SRPlat::SRMakeFinally([&] {
    if (!bWorkersFinished) {
      task.WaitComplete();
    }
    for (SRThreadCount i = 0; i < nSubtasks; i++) {
      pSubtasks[i].~taSubtask();
    }
  }); (void)subtasksFinally;

  while (nSubtasks < nWorkers) {
    new (pSubtasks + nSubtasks) taSubtask(&task, nSubtasks);
    nSubtasks++;
    _tpWorkers.Enqueue(pSubtasks + nSubtasks - 1);
  }

  bWorkersFinished = true; // Don't call again SRBaseTask::WaitComplete() if it throws here.
  task.WaitComplete();

  return task.TakeAggregateError(SRString::MakeUnowned("Failed " SR_FILE_LINE));
}

} // namespace ProbQA
