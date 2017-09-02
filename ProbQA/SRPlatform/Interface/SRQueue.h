// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRSimd.h"
#include "../SRPlatform/Interface/SRUtils.h"

namespace SRPlat {

// Items must be trivially copyable, constructible and copy-constructible. There can be non-trivial counterparts of the
//   above operations, but items must stay consistent after trivial operations executed by SRQueue.
template<typename taItem> class SRPLATFORM_API SRQueue {
private: // variables
  taItem *PTR_RESTRICT _pItems;
  size_t _iFirst;
  size_t _nItems : 56;
  size_t _logCapacity : 8;

private: // methods
  static size_t GetPaddedByteCount(const size_t nItems) {
    return SRSimd::PaddedBytesFromItems<sizeof(taItem)>(nItems);
  }
  ATTR_RESTRICT static taItem* ThrowingAlloc(const size_t nItems) {
    const size_t paddedBytes = GetPaddedByteCount(nItems);
    return reinterpret_cast<taItem*>(SRUtils::ThrowingSimdAlloc(paddedBytes));
  }

public: // methods
  explicit SRQueue(const uint8_t logInitialCapacity) {
    _logCapacity = logInitialCapacity;
    _pItems = ThrowingAlloc(size_t(1) << logInitialCapacity);
    _iFirst = 0;
    _nItems = 0;
  }
  void __vectorcall Push(taItem item) {
    size_t capacity = size_t(1) << _logCapacity;
    if(_nItems >= capacity) {
      assert(_nItems == capacity);
      capacity <<= 1;
      taItem *PTR_RESTRICT pNewItems = ThrowingAlloc(capacity);
      // The number of array head items is equal to _iFirst
      const size_t nTailItems = _nItems - _iFirst;
      //TODO: parametrize the class with taCache and put it as taCacheStore in 2 copy calls below
      // copy array tail (queue head)
      SRUtils::CopySaLu<true, false>(pNewItems, _pItems + _iFirst, sizeof(taItem) * nTailItems);
      // copy array head (queue tail)
      SRUtils::CopySuLa<true, false>(pNewItems + nTailItems, _pItems, sizeof(taItem) * _iFirst);
      // Ideally here we should just abandon the cache lines, without saving them to memory. But not bounds.
      SRUtils::FlushCache<true, true>(_pItems, _nItems * sizeof(taItem));
      _mm_free(_pItems);
      _pItems = pNewItems;
      // _nItems doesn't change yet
      _iFirst = 0;
      _logCapacity++;
    }
    const size_t iLim = (_iFirst + _nItems) & (capacity-1);
    _pItems[iLim] = item;
    _nItems++;
  }
  void Push(const taItem *const pSrc, const size_t nSrc) {
    size_t capacity = size_t(1) << _logCapacity;
    if(_nItems + nSrc > capacity) {
      const size_t newCapacity = capacity << 1;
      taItem *PTR_RESTRICT pNewItems = ThrowingAlloc(newCapacity);
      const size_t nTailItems = std::min(_nItems, capacity - _iFirst); // array tail (queue head)
      const size_t nHeadItems = _nItems - nTailItems; // array head (queue tail), if any
      //TODO: parametrize the class with taCache and put it as taCacheStore in 2 copy calls below
      SRUtils::CopySaLu<true, false>(pNewItems, _pItems + _iFirst, sizeof(taItem)*nTailItems);
      SRUtils::CopySuLa<true, false>(pNewItems + nTailItems, _pItems, sizeof(taItem)*nHeadItems);
      // Ideally here we should just abandon the cache lines, without saving them to memory. But not bounds.
      SRUtils::FlushCache<true, true>(_pItems, capacity * sizeof(taItem));
      _mm_free(_pItems);
      _pItems = pNewItems;
      // _nItems doesn't change yet
      _iFirst = 0;
      _logCapacity++;
      capacity = newCapacity;
    }
    //TODO: ensure that these translate into CMOVcc instructions: http://x86.renejeschke.de/html/file_module_x86_id_34.html
    const size_t nTail = std::min(nSrc, capacity - std::min(capacity, _iFirst + _nItems));
    const size_t nHead = nSrc - nTail;
    //TODO: parametrize the class with taCache and put it as taCacheStore in 2 copy calls below
    SRUtils::CopyUnalign<true,false>(_pItems + _iFirst + _nItems, pSrc, sizeof(taItem) * nTail);
    SRUtils::CopySaLu<true, false>(_pItems, pSrc + nTail, sizeof(taItem) * nHead);
    _nItems += nSrc;
  }
  taItem __vectorcall PopGet() {
    assert(_nItems >= 1);
    const size_t iRet = _iFirst;
    Pop();
    return _pItems[iRet];
  }
  void Pop() {
    _iFirst = (_iFirst + 1) & ((1 << _logCapacity) - 1);
    _nItems--;
  }
  taItem& Front() const {
    assert(_nItems >= 1);
    return _pItems[_iFirst];
  }
  size_t Size() const { return _nItems; }
};

} // namespace SRPlat
