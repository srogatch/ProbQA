// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRString.h"

namespace SRPlat {

class SRPLATFORM_API SRUtils {
public: // Methods
  static SRString PrintUtcTimestamp();
  static SRString PrintUtcDate();
  template<bool taSubmillisecond> SRPLATFORM_API static SRString PrintUtcTime();
  template<bool taSkipCache> SRPLATFORM_API static void FillZeroVects(__m256i *p, const size_t nVects);
};

} // namespace SRPlat
