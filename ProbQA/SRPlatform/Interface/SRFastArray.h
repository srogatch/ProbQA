#pragma once

#include "../SRPlatform/Interface/SRPlatform.h"
#include "../SRPlatform/Interface/SRMath.h"
#include "../SRPlatform/Interface/SRException.h"
#include "../SRPlatform/Interface/SRMessageBuilder.h"
#include "../SRPlatform/Interface/SRUtils.h"

namespace SRPlat {

class SRPLATFORM_API SRFastArrayBase {
protected: // variables
  size_t _count;
  SRFastArrayBase() : _count(0) {}
  explicit SRFastArrayBase(const size_t count) : _count(count) { }
};

// It must be possible to init and copy items trivially, but they may have constructors/assignment operators which do
//   not block this possibility.
template<typename taItem, bool taCacheDefault> class SRPLATFORM_API SRFastArray : public SRFastArrayBase {
public: // constants
  static constexpr uint8_t _cLogSimdBits = 8;

private: // variables
  taItem *_pItems;

private: // methods
  static size_t GetPaddedByteCount(const size_t nItems) {
    return SRMath::RoundUpToFactor(nItems * sizeof(taItem), sizeof(__m256i));
  }
  static taItem* ThrowingAllocBytes(const size_t paddedBytes) {
    taItem *pItems = reinterpret_cast<taItem*>(_mm_malloc(paddedBytes, sizeof(__m256i)));
    if (pItems == nullptr) {
      throw SRException(
        SRMessageBuilder("SRFastArray has failed to allocate ")(paddedBytes)(" bytes.").GetOwnedSRString());
    }
    return pItems;
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
    _mm_free(_pItems);
  }
  template<bool taFellowCD> SRFastArray(const SRFastArray<taItem, taFellowCD>& fellow) : SRFastArrayBase(fellow) {
    size_t paddedBytes;
    _pItems = ThrowingAlloc(_count, paddedBytes);
    const size_t nVects = paddedBytes >> (_cLogSimdBits - 3);
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
            (targetBytes)(" bytes."));
        }
      }
      _count = fellow._count;
      const size_t nVects = targetBytes >> (_cLogSimdBits - 3);
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

  //TODO: separate optimized method for sizeof(__m256i) == sizeof(taItem)
  template<bool taCache> typename std::enable_if_t<sizeof(__m256i) % sizeof(taItem) == 0>
  Fill(const taItem& item, size_t iStart, size_t iLim)
  {
    assert(iStart <= iLim);
    constexpr size_t cnItemsPerSimd = sizeof(__m256i) / sizeof(item);

    const __m256i vect = SRUtils::Set1(item);
    const size_t iVStart = SRMath::RoundUpToFactor(iStart, cnItemsPerSimd);
    if (iLim < iVStart) {
      memcpy(_pItems + iStart, &vect, (iLim - iStart) * sizeof(item));
      return;
    }

    __m256i *p = reinterpret_cast<__m256i *>(SRUtils::CopyPrologue<sizeof(item)>(_pItems + iStart, vect));
    assert(reinterpret_cast<void*>(p) == reinterpret_cast<void*>(_pItems + iVStart));
    __m256i *pLim = reinterpret_cast<__m256i*>(SRUtils::CopyEpilogue<sizeof(item)>(_pItems + iLim, vect));
    assert(pLim >= 2);

    size_t nVects = pLim - p;
    for (; nVects > 0; nVects--, p++) {
      taCache ? _mm256_store_si256(p, vect) : _mm256_stream_si256(p, vect);
    }
    if (!taCache) {
      _mm_sfence();
    }
  }
};

} // namespace SRPlat