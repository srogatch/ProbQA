// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "../SRPlatform/Interface/SRSpinSync.h"

namespace SRPlat {

namespace {
  std::atomic<uint64_t> gSpinSyncContention(0);
} // Anonymous namespace

uint64_t SRSpinStatistics::OnContention() {
  return 1 + gSpinSyncContention.fetch_add(1, std::memory_order_relaxed);
}

uint64_t SRSpinStatistics::TotalContention() {
  return gSpinSyncContention.load(std::memory_order_relaxed);
}

template class SRPLATFORM_API SRSpinSync<1<<5>;
template class SRPLATFORM_API SRSpinSync<1<<8>;

} // namespace SRPlat
