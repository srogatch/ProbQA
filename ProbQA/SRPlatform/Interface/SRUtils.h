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
