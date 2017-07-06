// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRCriticalSection.h"
#include "../SRPlatform/Interface/SRReaderWriterSync.h"

namespace SRPlat {

class SRPLATFORM_API SRConditionVariable {
  CONDITION_VARIABLE _block;
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
