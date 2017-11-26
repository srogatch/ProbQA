// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

namespace SRPlat {

template<typename taSync> class SRLock {
  taSync *_pSync;
public:
  SRLock() : _pSync(nullptr) {
  }
  explicit SRLock(taSync& sync) : _pSync(&sync) {
    _pSync->Acquire();
  }
  ~SRLock() {
    if (_pSync != nullptr) {
      _pSync->Release();
    }
  }
  SRLock(const SRLock&) = delete;
  SRLock& operator=(const SRLock&) = delete;
  void Init(taSync &sync) {
    assert(_pSync == nullptr);
    sync.Acquire();
    _pSync = &sync;
  }
  void EarlyRelease() {
    _pSync->Release();
    _pSync = nullptr;
  }
};

} // namespace SRPlat
