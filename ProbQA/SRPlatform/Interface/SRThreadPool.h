// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRCriticalSection.h"
#include "../SRPlatform/Interface/SRConditionVariable.h"
#include "../SRPlatform/Interface/ISRLogCustomizable.h"

namespace SRPlat {

class SRBaseSubtask;

class SRThreadPool : public ISRLogCustomizable {
  std::queue<SRBaseSubtask*> _qu;
  SRCriticalSection _cs;
  SRConditionVariable _haveWork;
  uint8_t _shutdownRequested : 1;

  //// Cache-insensitive data
  std::vector<std::thread> _workers;
  std::atomic<ISRLogger*> _pLogger;

private: // methods
  void WorkerEntry();

public:
  explicit SRThreadPool(const size_t nThreads = std::thread::hardware_concurrency());

  virtual ISRLogger* GetLogger() const override { return _pLogger.load(std::memory_order_relaxed); }
  virtual void SetLogger(ISRLogger *pLogger) override;
};

} // namespace SRPlat
