// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRSimd.h"
#include "../SRPlatform/Interface/SRNumTraits.h"

namespace SRPlat {

class SRPLATFORM_API SRVectMath {
  // The limit is 19 because we process only high 32 bits of doubles, and out of 20 bits of mantissa there, 1 bit is
  //   used for rounding.s
  static constexpr uint8_t _cnLog2TblBits = 10; // 1024 numbers times 8 bytes = 8KB.
  static const __m256d _c2divLn2; // 2.0/ln(2)
  static const __m256d _cDoubleNotExp;
  static const __m256d _cDoubleExp0;
  static const __m256i _cAvxExp2YMask;
  static const __m256d _cPlusBit;
  static const __m128i _cSseMantTblMask;
  static const __m128i _cExpNorm0;
  //static const __m128i _cSseRoundingMantTblMask;
  //static const __m128i _cSseRoundingBit;
  static double _plusLog2Table[1 << _cnLog2TblBits]; // plus |cnLog2TblBits|th highest bit
  static bool _isInitialized;

  static bool Initialize();
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
    
    // Clear the lowest exponent bit, so that exponent becomes 1022 indicating -1, for mantissas larger than sqrt(2).
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

  static __m256d __vectorcall Log2Hot(const __m256d x) {
    const __m256d zClearExp = _mm256_and_pd(_cDoubleNotExp, x);
    const __m256d z = _mm256_or_pd(zClearExp, _cDoubleExp0);

    //const __m128i high32 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(x), gHigh32Permute));
    const __m128 hiLane = _mm_castpd_ps(_mm256_extractf128_pd(x, 1));
    const __m128 loLane = _mm_castpd_ps(_mm256_castpd256_pd128(x));
    const __m128i high32 = _mm_castps_si128(_mm_shuffle_ps(loLane, hiLane, _MM_SHUFFLE(3, 1, 3, 1)));

    // This requires that x is non-negative, because the sign bit is not cleared before computing the exponent.
    const __m128i exps32 = _mm_srai_epi32(high32, SRNumTraits<double>::_cExponentOffs - 32);
    const __m128i normExps = _mm_sub_epi32(exps32, _cExpNorm0);

    // Compute y as approximately equal to log2(z)
    const __m128i indexes = _mm_and_si128(_cSseMantTblMask, _mm_srai_epi32(high32,
      SRNumTraits<double>::_cnMantissaBits - 32 - _cnLog2TblBits));
    //const __m256d y = _mm256_i32gather_pd(gPlusLog2Table, indexes, /*number of bytes per item*/ 8);
    const __m256d y = _mm256_set_pd(_plusLog2Table[indexes.m128i_u32[3]], _plusLog2Table[indexes.m128i_u32[2]],
      _plusLog2Table[indexes.m128i_u32[1]], _plusLog2Table[indexes.m128i_u32[0]]);
    // Compute A as z/exp2(y)
    const __m256d exp2_Y = _mm256_or_pd(_cPlusBit, _mm256_and_pd(z, _mm256_castsi256_pd(_cAvxExp2YMask)));

    // Calculate t=(A-1)/(A+1)
    const __m256d tNum = _mm256_sub_pd(z, exp2_Y);
    const __m256d tDen = _mm256_add_pd(z, exp2_Y); // both numerator and denominator would be divided by exp2_Y

    const __m256d t = _mm256_div_pd(tNum, tDen);

    const __m256d log2_z = _mm256_fmadd_pd(/*terms012345*/ t, _c2divLn2, y);

    const __m256d leading = _mm256_cvtepi32_pd(normExps); // leading integer part for the logarithm

    const __m256d log2_x = _mm256_add_pd(log2_z, leading);
    return log2_x;
  }
};

} // namespace SRPlat
