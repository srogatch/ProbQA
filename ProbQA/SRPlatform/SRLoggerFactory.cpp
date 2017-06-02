#include "stdafx.h"
#include "../SRPlatform/Interface/SRLoggerFactory.h"
#include "../SRPlatform/FileLogger.h"

namespace SRPlat {

ISRLogger* SRLoggerFactory::MakeFileLogger(const SRString& baseName) {
  return new FileLogger(baseName);
}

} // namespace SRPlat