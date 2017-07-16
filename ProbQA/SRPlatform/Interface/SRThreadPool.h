// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRBaseSubtask.h"
#include "../SRPlatform/Interface/SRBaseTask.h"
#include "../SRPlatform/Interface/SRCriticalSection.h"
#include "../SRPlatform/Interface/SRConditionVariable.h"

namespace SRPlat {

class SRThreadPool {
  std::vector<std::thread> _workers;
  std::queue<SRBaseSubtask*> _qu;
  SRCriticalSection _cs;
  SRConditionVariable _haveWork;
  uint8_t _shutdownRequested : 1;

private: // methods
  void WorkerEntry() {

  }
public:
  explicit SRThreadPool(const size_t nThreads) : _shutdownRequested(0) {
    _workers.reserve(nThreads);
    for (size_t i = 0; i < nThreads; i++) {
      _workers.emplace_back(&SRThreadPool::WorkerEntry, this);
    }
  }
};

} // namespace SRPlat
