// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"

TEST(SRBitArrayTest, Adding) {
  SRPlat::SRBitArray ba(1);
  uint64_t nBits[2] = { 1, 0 };
  for (int i = 1; i <= 1000; i++) {
    ba.Add(i >> 1, i & 1);
    nBits[i & 1] += i >> 1;
  }
  uint64_t sum = 0;
  for (uint64_t i = 0, iEn = ba.Size() >> 2; i<iEn; i++) {
    sum += __popcnt16(ba.GetQuad(i));
  }
  for (uint64_t i = ba.Size()&(~3ui64); i<ba.Size(); i++) {
    sum += ba.GetOne(i) ? 1 : 0;
  }
  EXPECT_EQ(nBits[1], sum);
  EXPECT_EQ(ba.Size(), nBits[0] + nBits[1]);
}
