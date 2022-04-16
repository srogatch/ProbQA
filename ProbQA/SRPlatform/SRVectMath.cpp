// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.
#include "../SRPlatform/Interface/SRVectMath.h"

namespace SRPlat {

const __m256d SRVectMath::_cdOne256 = _mm256_set1_pd(1.0);
const __m256d SRVectMath::_c2divLn2 = _mm256_set1_pd(2.8853900817779268147198493620038); // 2.0/ln(2)
const __m256d SRVectMath::_cLog2Coeff1 = _mm256_set1_pd(1.0 / 3);
const __m256d SRVectMath::_cLog2Coeff2 = _mm256_set1_pd(1.0 / 5);
const __m256d SRVectMath::_cDoubleNotExp = _mm256_castsi256_pd(_mm256_set1_epi64x(
  ~(SRNumTraits<double>::_cExponentMaskUp)));
const __m256d SRVectMath::_cDoubleExp0 = _mm256_castsi256_pd(_mm256_set1_epi64x(SRNumTraits<double>::_cExponent0Up));
const __m256d SRVectMath::_cAvxExp2YMask = _mm256_castsi256_pd(_mm256_set1_epi64x(
  ~((1ULL << (SRNumTraits<double>::_cExponentOffs - _cnLog2TblBits)) - 1)));
const __m256d SRVectMath::_cPlusBit = _mm256_castsi256_pd(_mm256_set1_epi64x(
  1ULL << (SRNumTraits<double>::_cExponentOffs - _cnLog2TblBits - 1)));
const __m128i SRVectMath::_cSseMantTblMask = _mm_set1_epi32((1 << _cnLog2TblBits) - 1);
const __m128i SRVectMath::_cExpNorm0 = _mm_set1_epi32(SRNumTraits<double>::_cExponent0Down);
//const __m128i SRVectMath::_cSseRoundingMantTblMask = _mm_set1_epi32(((1 << (_cnLog2TblBits + 1)) - 1)
//  << (SRNumTraits<double>::_cExponentOffs - 32 - _cnLog2TblBits - 1));
//const __m128i SRVectMath::_cSseRoundingBit = _mm_set1_epi32(
//  1 << (SRNumTraits<double>::_cExponentOffs - 32 - cnLog2TblBits - 1));
double SRVectMath::_plusLog2Table[1 << _cnLog2TblBits]; // plus |cnLog2TblBits|th highest bit

bool SRVectMath::_isInitialized = Initialize();

bool SRVectMath::Initialize() {
  constexpr double cCorr1 = 9.9999999999999927e-01; // Correction so that log2(1) <= 0
  for (uint32_t i = 0; i < (1 << _cnLog2TblBits); i++) {
    const uint64_t iZ = SRNumTraits<double>::_cExponent0Up 
      | (uint64_t(i) << (SRNumTraits<double>::_cnMantissaBits - _cnLog2TblBits));
    const double z = *reinterpret_cast<const double*>(&iZ);

    const uint64_t iZp = iZ | (1ULL << (SRNumTraits<double>::_cnMantissaBits - _cnLog2TblBits - 1));
    const double zp = *reinterpret_cast<const double*>(&iZp);
    const double l2zp = std::log2(zp);
    _plusLog2Table[i] = l2zp;
  }
  _plusLog2Table[0] *= cCorr1;
  return true;
}

} // namespace SRPlat
