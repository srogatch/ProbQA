// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "../SRPlatform/Interface/SRCriticalSection.h"

namespace SRPlat {

#if defined(_WIN32)
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
#elif defined(__unix__)
SRCriticalSection::SRCriticalSection() {
}

SRCriticalSection::SRCriticalSection(const uint32_t spinCount) {
  (void)spinCount;
}

SRCriticalSection::~SRCriticalSection() {
}

void SRCriticalSection::Acquire() {
  _mu.lock();
}

void SRCriticalSection::Release() {
  _mu.unlock();
}
#else
  #error "Unhandled OS"
#endif

} // namespace SRPlat
