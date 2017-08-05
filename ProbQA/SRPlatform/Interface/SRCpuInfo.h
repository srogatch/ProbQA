// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRPlatform.h"

namespace SRPlat {

//TODO: refactor from hard-coded values to values queried at runtime
class SRPLATFORM_API SRCpuInfo {
public:
  static const uint32_t _l1DataCachePerPhysCoreBytes = 32 * 1024;
  static const uint8_t _nLogicalCoresPerPhysCore = 2;
};

} // namespace SRPlat
