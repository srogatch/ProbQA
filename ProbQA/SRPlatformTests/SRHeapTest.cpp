// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"

using namespace SRPlat;

TEST(SRHeapTest, Down) {
  constexpr size_t cHeapSize = 1000;
  constexpr uint64_t cValLim = 2 * cHeapSize;
  SRFastRandom fr;
  SREntropyAdapter ea(fr);

  uint64_t heap[cHeapSize];
  for (size_t i = 2; i < cHeapSize; i++) {
    for (size_t j = 0; j < i; j++) {
      heap[j] = ea.Generate(cValLim);
    }
    std::make_heap(heap, heap + cHeapSize);
    heap[0] = ea.Generate(cValLim);
    SRHeapHelper::Down(heap, heap + cHeapSize);
    for (size_t j = 1; j < i; j++) {
      ASSERT_GE(heap[(j - 1) >> 1], heap[j]);
    }
  }
}
