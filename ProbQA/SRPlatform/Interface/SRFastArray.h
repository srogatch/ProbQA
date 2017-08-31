// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRAlignedDeleter.h"
#include "../SRPlatform/Interface/SRException.h"
#include "../SRPlatform/Interface/SRMath.h"
#include "../SRPlatform/Interface/SRMessageBuilder.h"
#include "../SRPlatform/Interface/SRPlatform.h"
#include "../SRPlatform/Interface/SRSimd.h"
#include "../SRPlatform/Interface/SRUtils.h"

namespace SRPlat {

class SRFastArrayBase {
protected: // variables
  size_t _count;
  SRFastArrayBase() : _count(0) {}
  explicit SRFastArrayBase(const size_t count) : _count(count) { }
};

// It must be possible to init and copy items trivially, but they may have constructors/assignment operators which do
//   not block this possibility.
template<typename taItem, bool taCacheDefault> class SRFastArray : public SRFastArrayBase {
public: // constants

private: // variables
  taItem *_pItems;

private: // methods
  //TODO: consider unifying such methods with SRQueue etc.
  static size_t GetPaddedByteCount(const size_t nItems) {
    return SRSimd::PaddedBytesFromItems<sizeof(taItem)>(nItems);
  }
  // Optimized version for the case when the item size is a power of 2 no larger than SIMD size. If it's larger than
  //   SIMD size, it would require left shitf rather than right shift for computing the number of vectors.
  static typename std::enable_if_t<sizeof(__m256i) % sizeof(taItem) == 0, size_t>
  GetNVects(const size_t nItems) {
    // Can't promote to a class variable because it's only valid for power-of-2 item sizes.
    static constexpr uint8_t logItemBytes = SRMath::StaticCeilLog2(sizeof(taItem));
    return SRMath::RShiftRoundUp(nItems, SRSimd::_cLogNBytes - logItemBytes);
  }
  static taItem* ThrowingAllocBytes(const size_t paddedBytes) {
    return reinterpret_cast<taItem*>(SRUtils::ThrowingSimdAlloc(paddedBytes));
  }
  static taItem* ThrowingAlloc(const size_t nItems, size_t& outPaddedBytes) {
    outPaddedBytes = GetPaddedByteCount(nItems);
    return ThrowingAllocBytes(outPaddedBytes);
  }
  static taItem* ThrowingAlloc(const size_t nItems) {
    size_t paddedBytes;
    return ThrowingAlloc(nItems, paddedBytes);
  }

public: // methods
  explicit SRFastArray() : _pItems(nullptr) { }
  explicit SRFastArray(const size_t count) : SRFastArrayBase(count), _pItems(ThrowingAlloc(count)) {
  }
  ~SRFastArray() {
    Clear();
  }
  template<bool taFellowCD> SRFastArray(const SRFastArray<taItem, taFellowCD>& fellow) : SRFastArrayBase(fellow) {
    size_t paddedBytes;
    _pItems = ThrowingAlloc(_count, paddedBytes);
    const size_t nVects = paddedBytes >> SRSimd::_cLogNBytes;
    SRUtils::Copy256<taCacheDefault, taFellowCD>(_pItems, fellow._pItems, nVects);
  }
  // Leaves the destination object empty if unable to allocate memory. This is to avoid excessive memory usage.
  template<bool taFellowCD> SRFastArray& operator=(const SRFastArray<taItem, taFellowCD>& fellow) {
    if (static_cast<SRFastArrayBase*>(this) != static_cast<const SRFastArrayBase*>(&fellow)) {
      const size_t oldBytes = GetPaddedByteCount(_count);
      const size_t targetBytes = GetPaddedByteCount(fellow._count);
      if (oldBytes != targetBytes) {
        _mm_free(_pItems);
        _pItems = _mm_malloc(targetBytes, sizeof(__m256i));
        if (_pItems == nullptr) {
          _count = 0;
          throw SRException(SRMessageBuilder(__FUNCTION__ " has failed to reallocate from ")(oldBytes)(" to ")
            (targetBytes)(" bytes.").GetOwnedSRString());
        }
      }
      _count = fellow._count;
      const size_t nVects = targetBytes >> SRSimd::_cLogNBytes;
      SRUtils::Copy256<taCacheDefault, taFellowCD>(_pItems, fellow._pItems, nVects);
    }
    return *this;
  }
  template<bool taFellowCD> SRFastArray(SRFastArray<taItem, taFellowCD>&& fellow)
    : SRFastArrayBase(std::forward<SRFastArrayBase>(fellow)), _pItems(fellow._pItems)
  {
    fellow._pItems = nullptr;
    fellow._count = 0;
  }
  template<bool taFellowCD> SRFastArray& operator=(SRFastArray<taItem, taFellowCD>&& fellow) {
    if (static_cast<SRFastArrayBase*>(this) != static_cast<const SRFastArrayBase*>(&fellow)) {
      _pItems = fellow._pItems;
      _count = fellow._count;
      fellow._pItems = nullptr;
      fellow._count = 0;
    }
    return *this;
  }

  // __vectorcall can pass in registers first 4 integer parameters, but 6 first vector parameters. Therefore vector
  //   parameters should be placed in the end.
  template<bool taCache> typename std::enable_if_t<
    sizeof(__m256i) % sizeof(taItem) == 0 && (sizeof(__m256i) > sizeof(taItem))> __vectorcall
  Fill(size_t iStart, size_t iLim, const taItem item) {
    assert(iStart <= iLim);
    // Can't promote to a class-level constant because it's only applicable when item size is a divisor of SIMD size.
    constexpr size_t cnItemsPerSimd = sizeof(__m256i) / sizeof(item);
    constexpr size_t cMaskIPS = cnItemsPerSimd - 1;
    static_assert((cnItemsPerSimd & cMaskIPS) == 0, "Number of items per SIMD must be a power of 2.");

    const __m256i vect = SRUtils::Set1(item);
    const size_t iVStart = (iStart + cMaskIPS) & (~cMaskIPS);
    if (iLim < iVStart) {
      memcpy(_pItems + iStart, &vect, (iLim - iStart) * sizeof(item));
      return;
    }

    __m256i *p = reinterpret_cast<__m256i *>(SRUtils::FillPrologue<sizeof(item)>(_pItems + iStart, vect));
    assert(reinterpret_cast<void*>(p) == reinterpret_cast<void*>(_pItems + iVStart));
    __m256i *pLim = reinterpret_cast<__m256i*>(SRUtils::FillEpilogue<sizeof(item)>(_pItems + iLim, vect));
    assert(pLim >= p && (((char*)pLim - (char*)p) & SRSimd::_cByteMask)== 0);

    size_t nVects = pLim - p;
    for (; nVects > 0; nVects--, p++) {
      taCache ? _mm256_store_si256(p, vect) : _mm256_stream_si256(p, vect);
    }
    if (!taCache) {
      _mm_sfence();
    }
  }

  // Optimized method for sizeof(__m256i) == sizeof(taItem)
  template<bool taCache> typename std::enable_if_t<sizeof(__m256i) == sizeof(taItem)> __vectorcall
  Fill(size_t iStart, size_t iLim, const taItem item) {
    assert(iStart <= iLim);
    size_t nVects = iLim - iStart;
    __m256i *p = reinterpret_cast<__m256i *>(_pItems + iStart);
    const __m256i& vect = *reinterpret_cast<const __m256i*>(&item);
    for (; nVects > 0; nVects--, p++) {
      taCache ? _mm256_store_si256(p, vect) : _mm256_stream_si256(p, vect);
    }
    if (!taCache) {
      _mm_sfence();
    }
  }

  template<bool taCache> typename std::enable_if_t<sizeof(__m256i) % sizeof(taItem) == 0> __vectorcall
  FillAll(const taItem item) {
    size_t nVects = GetNVects(_count);
    __m256i *p = reinterpret_cast<__m256i *>(_pItems);
    const __m256i vect = SRUtils::Set1(item);
    for (; nVects > 0; nVects--, p++) {
      taCache ? _mm256_store_si256(p, vect) : _mm256_stream_si256(p, vect);
    }
    if (!taCache) {
      _mm_sfence();
    }
  }

  // If newCount is greater than the current count, the new items are left uninitialized.
  // Note: unline vector::resize(), repeatedly calling this method results in quadratic complexity because the array
  //   doesn't operate any capacity, but only the size. This method is O(N).
  template<bool taCache> void Resize(const size_t newCount) {
    size_t newPaddedBytes;
    AlignedUniquePtr<taItem> pNewItems(ThrowingAlloc(newCount, newPaddedBytes));
    const size_t oldPaddedBytes = GetPaddedByteCount(_count);
    SRUtils::Copy256<taCache, false>(pNewItems.get(), _pItems,
      std::min(newPaddedBytes, oldPaddedBytes) >> SRSimd::_cLogNBytes);
    _mm_free(_pItems);
    _pItems = pNewItems.release();
    _count = newCount;
  }

  taItem& operator[](const size_t index) {
    return _pItems[index];
  }
  const taItem& operator[](const size_t index) const {
    return _pItems[index];
  }

  void Clear() {
    _mm_free(_pItems);
    _pItems = nullptr;
    _count = 0;
  }

  taItem* Get() {
    return _pItems;
  }
  taItem* Get() const {
    return _pItems;
  }
};

} // namespace SRPlat
