#include "stdafx.h"
#include "../SRPlatform/Interface/SRConditionVariable.h"

namespace SRPlat {

SRConditionVariable::SRConditionVariable() {
  InitializeConditionVariable(&_block);
}

SRConditionVariable::~SRConditionVariable() {
  // No deletion method in WinAPI for condition variable
}

bool SRConditionVariable::Wait(SRCriticalSection &cs, const uint32_t timeoutMiS) {
  return SleepConditionVariableCS(&_block, &(cs._block), timeoutMiS);
}

//TODO: bool Wait(SRReaderWriterSync &rws, const bool bLockShared, const uint32_t timeoutMiS = INFINITE);

void SRConditionVariable::WakeOne() {
  WakeConditionVariable(&_block);
}

void SRConditionVariable::WakeAll() {
  WakeAllConditionVariable(&_block);
}

} // namespace SRPlat