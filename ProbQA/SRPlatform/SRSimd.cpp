// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../SRPlatform/Interface/SRSimd.h"

namespace SRPlat {

const __m256i SRSimd::_cSet1MsbOffs = _mm256_set_epi32(32, 64, 96, 128, 160, 192, 224, 256);
const __m256i SRSimd::_cSet1LsbOffs = _mm256_set_epi32(256, 224, 192, 160, 128, 96, 64, 32);
const __m256i SRSimd::_cDoubleExpMaskUp = _mm256_set1_epi64x(SRNumTraits<double>::_cExponentMaskUp);
const __m256i SRSimd::_cDoubleExp0Up = _mm256_set1_epi64x(SRNumTraits<double>::_cExponent0Up);
const __m256i SRSimd::_cDoubleExp0Down = _mm256_set1_epi64x(SRNumTraits<double>::_cExponent0Down);

const __m128i SRSimd::_cDoubleExpMaskDown32 = _mm_set1_epi32(SRNumTraits<double>::_cExponentMaskDown);
const __m128i SRSimd::_cDoubleExp0Down32 = _mm_set1_epi32(SRNumTraits<double>::_cExponent0Down);

} // namespace SRPlat
