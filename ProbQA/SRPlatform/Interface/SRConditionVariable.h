#pragma once

#include "../SRPlatform/Interface/SRCriticalSection.h"

namespace SRPlat {

class SRPLATFORM_API SRConditionVariable {
  CONDITION_VARIABLE _block;
public:
  explicit SRConditionVariable();
  ~SRConditionVariable();
  bool Wait(SRCriticalSection &cs, const uint32_t timeoutMiS = INFINITE);
  //TODO: bool Wait(SRReaderWriterSync &rws, const bool bLockShared, const uint32_t timeoutMiS = INFINITE);
  void WakeOne();
  void WakeAll();
};

} // namespace SRPlat
