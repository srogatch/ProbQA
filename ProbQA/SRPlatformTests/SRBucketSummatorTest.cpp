// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"

using namespace SRPlat;

TEST(SRBucketSummator, Sequential) {
  static uint8_t memBss[SRBucketSummatorSeq<SRDoubleNumber>::GetMemoryRequirementBytes()];
  SRBucketSummatorSeq<SRDoubleNumber> bss(memBss);
  constexpr int64_t cnItems = 1000 * 1000;
  bss.ZeroBuckets();
  for (int64_t i = 0; i < cnItems; i++) {
    const __m256d nums = _mm256_set_pd(double(i), double(i) / cnItems, i / std::pow(cnItems, 2), i / pow(cnItems, 3));
    bss.CalcAdd(nums);
  }
  const __m256d neg = _mm256_set_pd(-1, -2, -3, -4);
  bss.CalcAdd(neg);

  const SRDoubleNumber sum = bss.ComputeSum();
  const double progression = 0.5 * (cnItems - 1.0) * cnItems;
  const double expected = progression / pow(cnItems, 3) + progression / std::pow(cnItems, 2) + progression / cnItems
    + progression - 10;
  ASSERT_NEAR(sum.GetValue(), expected, 1);
}
