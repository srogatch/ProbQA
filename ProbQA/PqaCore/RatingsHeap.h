// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/PqaCommon.h"

namespace ProbQA {

struct RatingsHeapItem {
  TPqaAmount _prob; // probability of the target at _iSource
  // Used as subtask index for heapify-based ListTopTargets(), or as a target index for RadixSort-based
  //   ListTopTargets(). The latter must use a sentinel item to mark where to stop fetching rated targets from that
  //   subtask's piece of targets.
  TPqaId _iSource;

  bool operator<(const RatingsHeapItem& fellow) const {
    return _prob < fellow._prob;
  }
};

} // namespace ProbQA
