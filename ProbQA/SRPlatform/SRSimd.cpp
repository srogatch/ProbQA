// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../SRPlatform/Interface/SRSimd.h"

namespace SRPlat {

const __m256i SRSimd::_cSet1MsbOffs = _mm256_set_epi32(32, 64, 96, 128, 160, 192, 224, 256);
const __m256i SRSimd::_cSet1LsbOffs = _mm256_set_epi32(256, 224, 192, 160, 128, 96, 64, 32);
const __m256i SRSimd::_cDoubleExpMask = _mm256_set1_epi64x(0x7ffULL << 52);
const __m256i SRSimd::_cDoubleExp0 = _mm256_set1_epi64x(1023ULL << 52);

} // namespace SRPlat
