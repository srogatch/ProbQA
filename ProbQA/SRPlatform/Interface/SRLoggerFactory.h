// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/ISRLogger.h"

namespace SRPlat {

class SRPLATFORM_API SRLoggerFactory {
public:
  static ISRLogger* MakeFileLogger(const SRString& baseName);
};

} // namespace SRPlat