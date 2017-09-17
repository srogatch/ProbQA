// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRMinimalTask.h"

namespace SRPlat {

template<typename taNumber> class SRBucketSummatorPar;

template<typename taNumber> class BucketerTask : public SRMinimalTask {
  SRBucketSummatorPar<taNumber> *_pBsp;

public: // variables
  const int64_t _iPartial; // index of the partial vector, or -1 if the last vector is full (i.e. no partial vector)
  const SRVectCompCount _nValid; // number of valid components in the partial vector

public:
  explicit BucketerTask(SRThreadPool &tp, SRBucketSummatorPar<taNumber> &bs, const int64_t iPartial,
    const SRVectCompCount nValid) : SRMinimalTask(tp), _pBsp(&bs), _iPartial(iPartial), _nValid(nValid)
  { }

  SRBucketSummatorPar<taNumber>& GetBucketSummator() const { return *_pBsp; }
};

} // namespace SRPlat
