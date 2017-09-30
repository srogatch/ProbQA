// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/PqaCommon.h"

namespace ProbQA {

struct RatingsHeapItem {
  RatedTarget _rt;
  TPqaId _iSource;

  bool operator<(const RatingsHeapItem& fellow) const {
    return _rt._prob < fellow._rt._prob;
  }
};

struct RatingsSourceInfo {
  TPqaId _iFirst;
  TPqaId _iLimit;
};

} // namespace ProbQA
