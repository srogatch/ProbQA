// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRCriticalSection.h"
#include "../SRPlatform/Interface/SRConditionVariable.h"
#include "../SRPlatform/Interface/ISRLogCustomizable.h"

namespace SRPlat {

class SRBaseSubtask;

class SRPLATFORM_API SRThreadPool : public ISRLogCustomizable {
public: // types
  // Returns |true| if worker thread should continue, or |false| if it should exit.
  typedef bool (*FCriticalCallback)(void *, SRException&&);

private: // variables
  std::queue<SRBaseSubtask*> _qu;
  SRCriticalSection _cs;
  SRConditionVariable _haveWork;
  uint8_t _shutdownRequested : 1;

  //// Cache-insensitive data
  std::vector<std::thread> _workers;
  std::atomic<ISRLogger*> _pLogger;
  FCriticalCallback _cbCritical;
  void *_pCcbData;

private: // methods
  void WorkerEntry();
  static bool DefaultCriticalCallback(void *pData, SRException &&ex);
  bool RunCriticalCallback(SRException &&ex);

public:
  explicit SRThreadPool(const size_t nThreads = std::thread::hardware_concurrency());

  virtual ISRLogger* GetLogger() const override { return _pLogger.load(std::memory_order_relaxed); }
  virtual void SetLogger(ISRLogger *pLogger) override;

  void SetCriticalCallback(FCriticalCallback f, void *pData);
};

} // namespace SRPlat
