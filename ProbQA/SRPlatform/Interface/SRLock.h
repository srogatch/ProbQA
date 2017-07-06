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
