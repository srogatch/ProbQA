// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRSimd.h"
#include "../SRPlatform/Interface/SRNumTraits.h"

namespace SRPlat {

class SRVectMath {
public:
  // For x<=0, a number smaller than -1023 is returned.
  static __m256d __vectorcall Log2Cold(const __m256d x) {
    // Clear exponent and sign bits
    const __m256d yClearExp = _mm256_and_pd(x, _mm256_set1_pd(SRCast::F64FromU64(
      /* exponent and sign bits 0, other bits 1*/
       ~(SRNumTraits<double>::_cExponentMaskUp | SRNumTraits<double>::_cSignMaskUp))));
    const __m256d yExp0 = _mm256_or_pd(yClearExp,
      _mm256_set1_pd(SRCast::F64FromU64(SRNumTraits<double>::_cExponent0Up)));

    const __m256d cmpResAvx = _mm256_cmp_pd(yExp0, /* sqrt(2) */ _mm256_set1_pd(1.4142135623730950488016887242097),
      _CMP_GT_OQ);
    
    // Clear the lowest exponent bit, so that exponent becomes 1022 indicating -1.
    const __m256d y = _mm256_xor_pd(yExp0, _mm256_and_pd(cmpResAvx, _mm256_set1_pd(
      /* Lowest exponent bit */ SRCast::F64FromU64(1ui64<< SRNumTraits<double>::_cExponentOffs) )));
    const __m128i cmpResSse = _mm_castps_si128(SRSimd::ExtractOdd(_mm256_castpd_ps(cmpResAvx)));

    // Calculate t=(y-1)/(y+1) and t**2
    const __m256d vd1 = _mm256_set1_pd(1.0);
    const __m256d tNum = _mm256_sub_pd(y, vd1);
    const __m256d tDen = _mm256_add_pd(y, vd1);
    const __m256d t = _mm256_div_pd(tNum, tDen);
    const __m256d t2 = _mm256_mul_pd(t, t); // t**2

    const __m256d t3 = _mm256_mul_pd(t, t2); // t**3
    const __m256d terms01 = _mm256_fmadd_pd(_mm256_set1_pd(1.0 / 3), t3, t);
    const __m256d t5 = _mm256_mul_pd(t3, t2); // t**5
    const __m256d terms012 = _mm256_fmadd_pd(_mm256_set1_pd(1.0 / 5), t5, terms01);
    const __m256d t7 = _mm256_mul_pd(t5, t2); // t**7
    const __m256d terms0123 = _mm256_fmadd_pd(_mm256_set1_pd(1.0 / 7), t7, terms012);
    const __m256d t9 = _mm256_mul_pd(t7, t2); // t**9
    const __m256d terms01234 = _mm256_fmadd_pd(_mm256_set1_pd(1.0 / 9), t9, terms0123);
    const __m256d t11 = _mm256_mul_pd(t9, t2); // t**11
    const __m256d terms012345 = _mm256_fmadd_pd(_mm256_set1_pd(1.0 / 11), t11, terms01234);

    const __m128i high32 = _mm_castps_si128(SRSimd::ExtractOdd(_mm256_castpd_ps(x)));
    // This requires that x is non-negative, because the sign bit is not cleared before computing the exponent.
    const __m128i exps32 = _mm_srai_epi32(high32, 20);

    // Set 1023 where cmpResSse is 0. Set 1022 where cmpResSse is 1.
    //const __m128i normalizer = _mm_xor_si128(_mm_set1_epi32(SRNumTraits<double>::_cExponent0Down),
    //  _mm_and_si128(_mm_set1_epi32(1), cmpResSse));
    //const __m128i normExps = _mm_sub_epi32(exps32, normalizer);
    const __m128i normExps = _mm_sub_epi32(exps32, _mm_add_epi32(
      _mm_set1_epi32(SRNumTraits<double>::_cExponent0Down), cmpResSse ));
    const __m256d expsPD = _mm256_cvtepi32_pd(normExps);

    const __m256d log2_x = _mm256_fmadd_pd(terms012345,
      /* 2.0/ln(2) */ _mm256_set1_pd(2.8853900817779268147208156228095), expsPD);
    return log2_x;
  }
};

} // namespace SRPlat
