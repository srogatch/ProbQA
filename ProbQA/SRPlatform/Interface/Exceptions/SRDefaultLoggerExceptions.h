#pragma once

#include "../SRPlatform/Interface/SRException.h"

namespace SRPlat {

class SRPLATFORM_API SRDefaultLoggerAlreadyInitializedException : public SRException {
public:
  SRDefaultLoggerAlreadyInitializedException(SRString &&message) : SRException(std::forward<SRString>(message)) { }
};

class SRPLATFORM_API SRDefaultLoggerConcurrentInitializationException : public SRException {
public:
  SRDefaultLoggerConcurrentInitializationException(SRString &&message) : SRException(std::forward<SRString>(message))
  { }
};

} // namespace SRPlat
