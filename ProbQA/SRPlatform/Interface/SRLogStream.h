#pragma once

#include "../SRPlatform/Interface/SRDefaultLogger.h"
#include "../SRPlatform/Interface/SRMessageBuilder.h"

namespace SRPlat {

class SRLogStream {
  SRMessageBuilder _mb;
  ISRLogger *_pLogger;
  ISRLogger::Severity _sev;
public:
  inline explicit SRLogStream(const ISRLogger::Severity sev, ISRLogger *pLogger = SRDefaultLogger::Get())
    : _sev(sev), _pLogger(pLogger) { }

  template<typename T> inline SRLogStream& operator<<(const T& arg) {
    _mb(arg);
    return *this;
  }

  inline ~SRLogStream() {
    _pLogger->Log(_sev, _mb.GetUnownedSRString());
  }
};

} // namespace SRPlat
