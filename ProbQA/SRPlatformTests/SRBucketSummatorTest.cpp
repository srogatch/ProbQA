// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"

using namespace SRPlat;

TEST(SRBucketSummator, Sequential) {
  alignas(SRSimd::_cNBytes) static uint8_t memBss[SRBucketSummatorSeq<SRDoubleNumber>::GetMemoryRequirementBytes()];
  SRBucketSummatorSeq<SRDoubleNumber> bss(memBss);
  constexpr int64_t cnItems = 1000 * 1000;

  bss.ZeroBuckets();

  const __m256d neg = _mm256_set_pd(-1, -2, -3, -4);
  bss.CalcAdd(neg);

  for (int64_t i = 0; i < cnItems; i++) {
    const __m256d nums = _mm256_set_pd(double(i), double(i) / cnItems, i / std::pow(cnItems, 2), i / pow(cnItems, 3));
    bss.CalcAdd(nums);
  }

  const SRDoubleNumber sum = bss.ComputeSum();
  const double progression = 0.5 * (cnItems - 1.0) * cnItems;
  const double expected = progression / pow(cnItems, 3) + progression / std::pow(cnItems, 2) + progression / cnItems
    + progression - 10;
  ASSERT_NEAR(sum.GetValue(), expected, 1e-3);
}

TEST(SRBucketSummator, Parallel) {
  constexpr int32_t cnWorkers = 16;
  SRThreadPool tp(cnWorkers, 0);
  alignas(SRSimd::_cNBytes) static uint8_t memBsp[
    SRBucketSummatorPar<SRDoubleNumber>::GetMemoryRequirementBytes(cnWorkers)];
  SRBucketSummatorPar<SRDoubleNumber> bsp(cnWorkers, memBsp);
  alignas(SRSimd::_cNBytes) static uint8_t memSubtasks[
    SRBucketSummatorPar<SRDoubleNumber>::_cSubtaskMemReq * cnWorkers];
  SRPoolRunner pr(tp, memSubtasks);
  constexpr int64_t cnItems = 1000 * 1000;

  const __m256d neg = _mm256_set_pd(-1, -2, -3, -4);
  for (int32_t i = 0; i < cnWorkers; i++) {
    bsp.ZeroBuckets(i);
    bsp.CalcAdd(i, neg);
  }

  for (int64_t i = 0; i < cnItems; i++) {
    const __m256d nums = _mm256_set_pd(double(i), double(i) / cnItems, i / std::pow(cnItems, 2), i / pow(cnItems, 3));
    bsp.CalcAdd(i%cnWorkers, nums);
  }

  const SRDoubleNumber sum = bsp.ComputeSum(pr);
  const double progression = 0.5 * (cnItems - 1.0) * cnItems;
  const double expected = progression / pow(cnItems, 3) + progression / std::pow(cnItems, 2) + progression / cnItems
    + progression - 10 * cnWorkers;
  ASSERT_NEAR(sum.GetValue(), expected, 1e-3);
}