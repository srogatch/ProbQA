// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../SRPlatform/Interface/SRDoubleNumber.h"

namespace SRPlat {

const __m128i SRDoubleNumber::_cSizeBytes128_32 = _mm_set1_epi32(sizeof(SRDoubleNumber));
const SRPacked64 SRDoubleNumber::_cSizeBytes64_32 = SRPacked64::Set1U32(sizeof(SRDoubleNumber));

} // namespace SRPlat
