// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"

using namespace SRPlat;

TEST(SRVectMathTest, Log2Cold) {
  SRFastRandom fr;
  __m256d numsF64 = _mm256_set_pd(-3, -2, -1, 0);
  {
    const __m256d actual = SRVectMath::Log2Cold(numsF64);
    for (int8_t j = 0; j <= 3; j++) {
      ASSERT_LE(actual.m256d_f64[j], -1022.5);
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
        ASSERT_LE(actual.m256d_f64[j], -1022.5);
        continue;
      }
      const double expected = std::log2(numsF64.m256d_f64[j]);
      const double absErr = std::fabs(expected * 1e-12);
      ASSERT_NEAR(expected, actual.m256d_f64[j], absErr);
    }
  }
}

TEST(SRVectMathTest, Log2Hot) {
  SRFastRandom fr;
  __m256d numsF64 = _mm256_set_pd(-3, -2, -1, 0);
  {
    const __m256d actual = SRVectMath::Log2Hot(numsF64);
    for (int8_t j = 0; j <= 3; j++) {
      ASSERT_LE(actual.m256d_f64[j], -1022.5);
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
        ASSERT_LE(actual.m256d_f64[j], -1022.5);
        continue;
      }
      const double expected = std::log2(numsF64.m256d_f64[j]);
      const double absErr = std::fabs(expected * 1e-12);
      ASSERT_NEAR(expected, actual.m256d_f64[j], absErr);
    }
  }
}