#include "stdafx.h"
#include "../SRPlatform/Interface/SRConditionVariable.h"
#include "../SRPlatform/Interface/SRMacros.h"

namespace SRPlat {

SRConditionVariable::SRConditionVariable() {
  InitializeConditionVariable(&_block);
}

SRConditionVariable::~SRConditionVariable() {
  // No deletion method in WinAPI for condition variable
}

bool SRConditionVariable::Wait(SRCriticalSection &cs, const uint32_t timeoutMiS) {
  bool answer = SleepConditionVariableCS(&_block, &(cs._block), timeoutMiS);
  if (!answer) {
    uint32_t le = GetLastError();
    if (le != ERROR_TIMEOUT) {
      // Can't use file logger because it depends on SRConditionVariable
      SR_LOG_WINFAIL(Critical, SRDefaultLogger::Dbg(), le);
    }
  }
  return answer;
}

bool SRConditionVariable::Wait(SRReaderWriterSync &rws, const bool bLockExclusive, const uint32_t timeoutMiS) {
  bool answer = SleepConditionVariableSRW(&_block, &(rws._block), timeoutMiS,
    bLockExclusive ? 0 : CONDITION_VARIABLE_LOCKMODE_SHARED);
  if (!answer) {
    uint32_t le = GetLastError();
    if (le != ERROR_TIMEOUT) {
      // Can't use file logger because it depends on SRConditionVariable
      SR_LOG_WINFAIL(Critical, SRDefaultLogger::Dbg(), le);
    }
  }
  return answer;
}

void SRConditionVariable::WakeOne() {
  WakeConditionVariable(&_block);
}

void SRConditionVariable::WakeAll() {
  WakeAllConditionVariable(&_block);
}

} // namespace SRPlat