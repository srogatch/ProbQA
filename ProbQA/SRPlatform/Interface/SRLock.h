// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

namespace SRPlat {

template<typename taSync> class SRLock {
  taSync *_pSync;
public:
  explicit SRLock(taSync& sync) : _pSync(&sync) {
    _pSync->Acquire();
  }
  void EarlyRelease() {
    _pSync->Release();
    _pSync = nullptr;
  }
  ~SRLock() {
    if (_pSync) {
      _pSync->Release();
    }
  }
};

} // namespace SRPlat
