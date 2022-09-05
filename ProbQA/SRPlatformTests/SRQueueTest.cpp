// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"

using namespace SRPlat;

TEST(SRQueue, Intense) {
  SRFastRandom fr;
  const int64_t cnItems = 1000 * 1000;
  SRQueue<uint64_t> tested(0);
  std::queue<uint64_t> reference;

  for (int64_t i = 0; i < cnItems; i++) {
    bool bPush = true;
    if (reference.size() > 0) {
      if (fr.Generate64() % 3 == 0) {
        bPush = false;
      }
    }
    if (bPush) {
      const uint64_t item = fr.Generate64();
      tested.Push(item);
      reference.push(item);
    }
    else {
      const uint64_t expected = reference.front();
      reference.pop();
      const uint64_t actual = tested.PopGet();
      ASSERT_EQ(actual, expected);
    }
  }
}
