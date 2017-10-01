// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

namespace SRPlat {

class SRHeapHelper {
public:
  // After update of its key, let the former heap top item find its place down the heap.
  template<typename taItem> inline static void Down(taItem *pFirst, taItem *pLimit);
};

template<typename taItem> inline void SRHeapHelper::Down(taItem *PTR_RESTRICT pFirst, taItem *PTR_RESTRICT pLimit) {
  taItem *PTR_RESTRICT pCur = pFirst;
  for (;;) {
    const size_t curAt = pCur - pFirst;
    taItem *PTR_RESTRICT pChild1 = pFirst + 2 * curAt + 1;
    if (pChild1 >= pLimit) {
      return;
    }
    taItem *PTR_RESTRICT pChild2 = pChild1 + 1;
    if (pChild2 >= pLimit) {
      if (*pCur < *pChild1) {
        std::swap(*pCur, *pChild1);
      }
      return;
    }
    taItem *PTR_RESTRICT pHigherChild = ((*pChild2 < *pChild1) ? pChild1 : pChild2);
    if (!(*pCur < *pHigherChild)) {
      return;
    }
    std::swap(*pCur, *pHigherChild);
    pCur = pHigherChild;
  }
}

} // namespace SRPlat
