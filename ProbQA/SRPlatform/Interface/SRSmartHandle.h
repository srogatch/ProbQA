#pragma once

namespace SRPlat {

typedef HANDLE SRHandle;

template <SRHandle taEmpty> class SRSmartHandle {
  SRHandle _srh;
public:
  explicit SRSmartHandle(SRHandle srh = taEmpty) : _srh(srh) {
  }
  ~SRSmartHandle() {
    if (_srh != taEmpty) {
      if (!CloseHandle(_srh)) {
        //TODO: GetLastError() and log it, because we shouldn't throw exceptions from destructors
      }
    }
  }
};

} // namespace SRPlat