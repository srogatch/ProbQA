// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRCriticalSection.h"
#include "../SRPlatform/Interface/SRReaderWriterSync.h"

namespace SRPlat {

class SRPLATFORM_API SRConditionVariable {
#if defined(_WIN32)
  CONDITION_VARIABLE _block;
#elif defined(__unix__)
  std::condition_variable _cv;
#else
  #error "Unhandled OS"
#endif // OS
public:
  explicit SRConditionVariable();
  ~SRConditionVariable();

  SRConditionVariable(const SRConditionVariable&) = delete;
  SRConditionVariable& operator=(const SRConditionVariable&) = delete;
  SRConditionVariable(SRConditionVariable&&) = delete;
  SRConditionVariable& operator=(SRConditionVariable&&) = delete;

  bool Wait(SRCriticalSection &cs, const uint32_t timeoutMiS = INFINITE);
  bool Wait(SRReaderWriterSync &rws, const bool bLockExclusive, const uint32_t timeoutMiS = INFINITE);
  void WakeOne();
  void WakeAll();
};

} // namespace SRPlat
