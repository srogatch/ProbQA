#include "stdafx.h"
#include "../SRPlatform/Interface/SRCriticalSection.h"

namespace SRPlat {

SRCriticalSection::SRCriticalSection() {
  InitializeCriticalSection(&_block);
}

SRCriticalSection::SRCriticalSection(const uint32_t spinCount) {
  InitializeCriticalSectionAndSpinCount(&_block, spinCount);
}

SRCriticalSection::~SRCriticalSection() {
  DeleteCriticalSection(&_block);
}

void SRCriticalSection::Acquire() {
  EnterCriticalSection(&_block);
}

void SRCriticalSection::Release() {
  LeaveCriticalSection(&_block);
}

} // namespace SRPlat
