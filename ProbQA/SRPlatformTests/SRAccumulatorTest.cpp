// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"

using namespace SRPlat;

namespace SRTest {
  class SRAccumVectDbl256TestHelper {
  public:
    static void __vectorcall SetState(SRAccumVectDbl256& vectAcc, const __m256d sum, const __m256d corr) {
      vectAcc._sum = sum;
      vectAcc._corr = corr;
    }
  };
}

using namespace SRTest;

TEST(SRAccumVectDbl256, PairSum) {
  SRAccumVectDbl256 va1;
  SRAccumVectDbl256TestHelper::SetState(va1, _mm256_set_pd(128, 64, 32, 16), _mm256_set_pd(8, 4, 2, 1));
  EXPECT_EQ(va1.PreciseSum(), 225);
  EXPECT_EQ(va1.GetFullSum(), 225);
  SRAccumVectDbl256 va2;
  SRAccumVectDbl256TestHelper::SetState(va2, _mm256_set_pd(32768, 16384, 8192, 4096),
    _mm256_set_pd(2048, 1024, 512, 256));
  double s2;
  double s1 = va1.PairSum(va2, s2);
  EXPECT_EQ(s1, 225);
  EXPECT_EQ(s2, 57600);
  EXPECT_EQ(va2.GetFullSum(), 57600);
  EXPECT_EQ(va2.PreciseSum(), 57600);
}
