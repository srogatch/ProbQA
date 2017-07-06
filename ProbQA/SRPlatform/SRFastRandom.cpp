// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../SRPlatform/Interface/SRFastRandom.h"

namespace SRPlat {

const __m128i SRFastRandom::_cRShift = _mm_set_epi64x(26, 17);

} // namespace SRPlat
