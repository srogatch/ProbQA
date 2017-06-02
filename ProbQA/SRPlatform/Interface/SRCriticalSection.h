#pragma once

#include "../SRPlatform/Interface/SRPlatform.h"

namespace SRPlat {

class SRPLATFORM_API SRCriticalSection {
  friend class SRConditionVariable;
  CRITICAL_SECTION _block;
public:
  explicit SRCriticalSection();
  explicit SRCriticalSection(const uint32_t spinCount);
  ~SRCriticalSection();
  void Acquire(); // Enter
  void Release(); // Leave
};

} // namespace SRPlat
