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
  struct RareData; // cache-insensitive piece of thread pool data

public: // types
  // Returns |true| if worker thread should continue, or |false| if it should exit.
  typedef bool (*FCriticalCallback)(void *, SRException&&);

private: // variables
  std::queue<SRBaseSubtask*> _qu;
  SRCriticalSection _cs;
  SRConditionVariable _haveWork;
  uint8_t _shutdownRequested : 1;
  RareData *_pRd;

private: // methods
  void WorkerEntry();
  static bool DefaultCriticalCallback(void *pData, SRException &&ex);
  bool RunCriticalCallback(SRException &&ex);

public:
  explicit SRThreadPool(const size_t nThreads = std::thread::hardware_concurrency());
  virtual ~SRThreadPool() override;

  virtual ISRLogger* GetLogger() const override;
  virtual void SetLogger(ISRLogger *pLogger) override;

  // If f==nullptr , the function sets the default callback with |this| as data.
  void SetCriticalCallback(FCriticalCallback f, void *pData = nullptr);
};

} // namespace SRPlat
