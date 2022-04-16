// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "../SRPlatform/Interface/SRLoggerFactory.h"
#include "../SRPlatform/FileLogger.h"

namespace SRPlat {

ISRLogger* SRLoggerFactory::MakeFileLogger(const SRString& baseName) {
  return new FileLogger(baseName);
}

} // namespace SRPlat