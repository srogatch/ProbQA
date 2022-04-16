// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "../SRPlatform/Interface/SRReaderWriterSync.h"

namespace SRPlat {

template<> SRPLATFORM_API void SRReaderWriterSync::Acquire<true>() {
  AcquireSRWLockExclusive(&_block);
}

template<> SRPLATFORM_API void SRReaderWriterSync::Acquire<false>() {
  AcquireSRWLockShared(&_block);
}

template<> SRPLATFORM_API bool SRReaderWriterSync::TryAcquire<true>() {
  return TryAcquireSRWLockExclusive(&_block);
}

template<> SRPLATFORM_API bool SRReaderWriterSync::TryAcquire<false>() {
  return TryAcquireSRWLockShared(&_block);
}

template<> SRPLATFORM_API void SRReaderWriterSync::Release<true>() {
  ReleaseSRWLockExclusive(&_block);
}

template<> SRPLATFORM_API void SRReaderWriterSync::Release<false>() {
  ReleaseSRWLockShared(&_block);
}

void SRReaderWriterSync::Acquire(const bool bExclusive) {
  bExclusive ?  Acquire<true>() : Acquire<false>();
}

bool SRReaderWriterSync::TryAcquire(const bool bExclusive) {
  return (bExclusive ? TryAcquire<true>() : TryAcquire<false>());
}

void SRReaderWriterSync::Release(const bool bExclusive) {
  bExclusive ? Release<true>() : Release<false>();
}

} // namespace SRPlat
