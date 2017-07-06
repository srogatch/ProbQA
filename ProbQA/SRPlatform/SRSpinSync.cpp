// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
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

} // namespace SRPlat