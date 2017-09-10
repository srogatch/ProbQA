// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRMacros.h"
#include "../SRPlatform/Interface/SRRealNumber.h"

namespace SRPlat {

class SRNumHelper {
public:
  // Returns the number of vectors.
  template<typename taNumber> ATTR_NOALIAS static int64_t
  Vectorize(const int64_t nItems, int64_t &PTR_RESTRICT iPartial, SRVectCompCount &PTR_RESTRICT nValid)
  {
    constexpr bool isPowOf2 = SRMath::StaticIsPowOf2(SRNumPack<taNumber>::_cnComps);
    if (isPowOf2) {
      constexpr SRVectCompCount shift = SRMath::StaticFloorLog2(SRNumPack<taNumber>::_cnComps);
      nValid = nItems & ((1 << shift) - 1);
      const int64_t quot = nItems >> shift;
      iPartial = (nValid ? quot : -2);
      return quot + (nValid ? 1 : 0);
    }
    else {
      constexpr SRVectCompCount nPerVect = SRNumPack<taNumber>::_cnComps;
      const lldiv_t divRes = lldiv(nItems, nPerVect);
      nValid = static_cast<SRVectCompCount>(divRes.rem);
      iPartial = (nValid ? divRes.quot : -2);
      return divRes.quot + (nValid ? 1 : 0);
    }
  }
};

} // namespace SRPlat
