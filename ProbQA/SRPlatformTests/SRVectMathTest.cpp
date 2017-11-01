// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"

using namespace SRPlat;

TEST(SRVectMathTest, Log2Cold) {
  SRFastRandom fr;
  __m256d numsF64;
  { // Test exactly 1.0
    numsF64 = _mm256_set_pd(0.999, 0.9999, 0.99999, 1);
    const __m256d actual = SRVectMath::Log2Cold(numsF64);
    for (int8_t j = 0; j <= 3; j++) {
      EXPECT_LE(actual.m256d_f64[j], 0);
    }
    EXPECT_GE(actual.m256d_f64[0], -1e-19);
  }
  { // Test negatives and 0
    numsF64 = _mm256_set_pd(-3, -2, -1, 0);
    const __m256d actual = SRVectMath::Log2Cold(numsF64);
    for (int8_t j = 0; j <= 3; j++) {
      EXPECT_LE(actual.m256d_f64[j], -1022.5);
    }
  }
  for (int64_t i = 0; i < 1000 * 1000; i++) {
    const __m256i numsI64 = fr.Generate<__m256i>();
    for (int8_t j = 0; j <= 3; j++) {
      numsF64.m256d_f64[j] = double(numsI64.m256i_u64[j]);
    }
    const __m256d actual = SRVectMath::Log2Cold(numsF64);
    for (int8_t j = 0; j <= 3; j++) {
      if (numsI64.m256i_u64[j] <= 0) {
        EXPECT_LE(actual.m256d_f64[j], -1022.5);
        continue;
      }
      const double expected = std::log2(numsF64.m256d_f64[j]);
      const double absErr = std::fabs(expected * 1e-12);
      EXPECT_NEAR(expected, actual.m256d_f64[j], absErr);
    }
  }
}

TEST(SRVectMathTest, Log2Hot) {
  constexpr double reqPrec = 3e-12; // required relative precision
  SRFastRandom fr;
  __m256d numsF64;
  { // Test exactly 1.0
    numsF64 = _mm256_set_pd(0.999, 0.9999, 0.99999, 1);
    const __m256d actual = SRVectMath::Log2Hot(numsF64);
    for (int8_t j = 0; j <= 3; j++) {
      EXPECT_LE(actual.m256d_f64[j], 0);
    }
    EXPECT_GE(actual.m256d_f64[0], -reqPrec);
  }
  { // Test negatives and 0
    numsF64 = _mm256_set_pd(-3, -2, -1, 0);
    const __m256d actual = SRVectMath::Log2Hot(numsF64);
    for (int8_t j = 0; j <= 3; j++) {
      EXPECT_LE(actual.m256d_f64[j], -1022.5);
    }
  }

  const int32_t expRange = 2046;
  const uint8_t nMantBits = 22;
  for (uint64_t i = 0; i < (1<<nMantBits); i+=4) {
    for (int8_t j = 0; j <= 3; j++) {
      const int32_t curExp = ((i+j) % expRange) + 1; // 1 to 2046 without offset correspond to -1022 to 1023
      const uint64_t iX = (uint64_t(curExp) << SRNumTraits<double>::_cExponentOffs)
        | ((i+j) << (SRNumTraits<double>::_cMantissaOffs + SRNumTraits<double>::_cnMantissaBits - nMantBits));
      numsF64.m256d_f64[j] = SRCast::F64FromU64(iX);
    }
    const __m256d actual = SRVectMath::Log2Hot(numsF64);
    for (int8_t j = 0; j <= 3; j++) {
      if (numsF64.m256d_f64[j] == 1) {
        continue; // must be already tested separately
      }
      const double expected = std::log2(numsF64.m256d_f64[j]);
      const double absErr = std::fabs(expected * reqPrec);
      EXPECT_NEAR(expected, actual.m256d_f64[j], absErr);
    }
  }

  for (int64_t i = 0; i < 1000 * 1000; i++) {
    const __m256i numsI64 = fr.Generate<__m256i>();
    for (int8_t j = 0; j <= 3; j++) {
      numsF64.m256d_f64[j] = double(numsI64.m256i_u64[j]);
    }
    const __m256d actual = SRVectMath::Log2Hot(numsF64);
    for (int8_t j = 0; j <= 3; j++) {
      if (numsI64.m256i_u64[j] <= 0) {
        EXPECT_LE(actual.m256d_f64[j], -1022.5);
        continue;
      }
      if (numsF64.m256d_f64[j] == 1) {
        continue; // must be already tested separately
      }
      const double expected = std::log2(numsF64.m256d_f64[j]);
      const double absErr = std::fabs(expected * reqPrec);
      EXPECT_NEAR(expected, actual.m256d_f64[j], absErr);
    }
  }
}