#include "stdafx.h"
#include "../SRPlatform/Interface/SRReaderWriterSync.h"

namespace SRPlat {

template<> void SRReaderWriterSync::Acquire<true>() {
  AcquireSRWLockExclusive(&_block);
}

template<> void SRReaderWriterSync::Acquire<false>() {
  AcquireSRWLockShared(&_block);
}

template<> bool SRReaderWriterSync::TryAcquire<true>() {
  return TryAcquireSRWLockExclusive(&_block);
}

template<> bool SRReaderWriterSync::TryAcquire<false>() {
  return TryAcquireSRWLockShared(&_block);
}

template<> void SRReaderWriterSync::Release<true>() {
  ReleaseSRWLockExclusive(&_block);
}

template<> void SRReaderWriterSync::Release<false>() {
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
