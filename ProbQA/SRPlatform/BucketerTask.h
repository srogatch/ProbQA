// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRMinimalTask.h"

namespace SRPlat {

template<typename taNumber> class SRBucketSummator;

template<typename taNumber> class BucketerTask : public SRMinimalTask {
  SRBucketSummator<taNumber> *_pBs;

public: // variables
  const int64_t _iPartial; // index of the partial vector, or -1 if the last vector is full (i.e. no partial vector)
  const SRVectCompCount _nValid; // number of valid components in the partial vector

public:
  explicit BucketerTask(SRThreadPool &tp, SRBucketSummator<taNumber> &bs, const int64_t iPartial,
    const SRVectCompCount nValid) : SRMinimalTask(tp), _pBs(&bs), _iPartial(iPartial), _nValid(nValid)
  { }

  SRBucketSummator<taNumber>& GetBS() const { return *_pBs; }
};

} // namespace SRPlat
