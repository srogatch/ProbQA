#pragma once

#include "../SRPlatform/Interface/ISRLogger.h"

namespace SRPlat {

class SRPLATFORM_API SRLoggerFactory {
public:
  static ISRLogger* MakeFileLogger(const SRString& baseName);
};

} // namespace SRPlat