// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRPlatform.h"

namespace SRPlat {

//TODO: refactor from hard-coded values to values queried at runtime
// Get cache line size: https://stackoverflow.com/questions/794632/programmatically-get-the-cache-line-size
class SRPLATFORM_API SRCpuInfo {
public:
  static constexpr uint32_t _l1DataCachePerPhysCoreBytes = 32 * 1024;
  static constexpr uint8_t _logCacheLineBytes = 6;
  static constexpr uint8_t _nLogicalCoresPerPhysCore = 2;
  static constexpr uint16_t _cacheLineBytes = 1 << _logCacheLineBytes;
  static constexpr uintptr_t _cacheLineMask = _cacheLineBytes - 1;
};

} // namespace SRPlat
