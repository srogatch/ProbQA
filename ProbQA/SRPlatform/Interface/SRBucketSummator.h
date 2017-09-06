// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRThreadPool.h"

namespace SRPlat {

template<typename taNumber> class SRBucketSummator {
public:
  static size_t GetMemoryRequirementBytes(const SRThreadPool::TThreadCount nWorkers);
};

} // namespace SRPlat
