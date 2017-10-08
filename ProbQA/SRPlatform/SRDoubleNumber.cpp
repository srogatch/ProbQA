// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../SRPlatform/Interface/SRDoubleNumber.h"

namespace SRPlat {

__m128i __vectorcall SRDoubleNumber::ScaleBySizeBytesU32(const __m128i a) {
  static_assert(sizeof(SRDoubleNumber) == (1 << 3), "Hard-coded below");
  // When the size is not a power of 2, _mm_mullo_epi32() can be used.
  const __m128i ans = _mm_slli_epi32(a, 3);
  return ans;
}

} // namespace SRPlat
