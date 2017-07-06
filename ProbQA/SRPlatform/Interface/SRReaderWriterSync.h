// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRPlatform.h"

namespace SRPlat {

class SRPLATFORM_API SRReaderWriterSync {
  friend class SRConditionVariable;
  SRWLOCK _block = SRWLOCK_INIT;
public:
  explicit SRReaderWriterSync() { }
  ~SRReaderWriterSync() { }

  SRReaderWriterSync(const SRReaderWriterSync&) = delete;
  SRReaderWriterSync& operator=(const SRReaderWriterSync&) = delete;
  SRReaderWriterSync(SRReaderWriterSync&&) = delete;
  SRReaderWriterSync& operator=(SRReaderWriterSync&&) = delete;

  template<bool taExclusive> SRPLATFORM_API void Acquire();
  template<bool taExclusive> SRPLATFORM_API bool TryAcquire();
  template<bool taExclusive> SRPLATFORM_API void Release();

  void Acquire(const bool bExclusive);
  bool TryAcquire(const bool bExclusive);
  void Release(const bool bExclusive);
};

template<bool taExclusive> class SRRWLock {
  SRReaderWriterSync *_pRws;
public:
  explicit SRRWLock(SRReaderWriterSync& rws) : _pRws(&rws) {
    _pRws->Acquire<taExclusive>();
  }
  void EarlyRelease() {
    _pRws->Release<taExclusive>();
    _pRws = nullptr;
  }
  ~SRRWLock() {
    if (_pRws != nullptr) {
      _pRws->Release<taExclusive>();
    }
  }
};

} // namespace SRPlat
