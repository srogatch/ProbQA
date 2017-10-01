// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRPlatform.h"
#include "../SRPlatform/Interface/SRMacros.h"
#include "../SRPlatform/Interface/SRException.h"
#include "../SRPlatform/Interface/SRMessageBuilder.h"
#include "../SRPlatform/Interface/SRSimd.h"

namespace SRPlat {

class SRPLATFORM_API SRBaseMemPool {
public:
  virtual ~SRBaseMemPool() { }
  virtual void FreeAllChunks() { }
  ATTR_RESTRICT virtual void* AllocMem(const size_t nBytes) {
    return _mm_malloc(SRSimd::GetPaddedBytes(nBytes), SRSimd::_cNBytes);
  }
  virtual void ReleaseMem(void *PTR_RESTRICT p, const size_t) { _mm_free(p); }
};

SRPLATFORM_API SRBaseMemPool& SRGetBaseMemPool();

// If the unit is 256-bit, then taLogUnitBits should be 8 because 256 == (1<<8) .
// This class is thread-safe, except some methods explicitly specified as not thread-safe.
template<uint32_t taLogNUnitBits, uint32_t taNGranules> class SRMemPool : public SRBaseMemPool {
  static_assert(taLogNUnitBits >= 3, "Must be integer number of bytes.");

public: // constants
  static const size_t _cLogNUnitBytes = taLogNUnitBits - 3;
  static const size_t _cNUnitBytes = 1 << _cLogNUnitBytes;

  static_assert(SRSimd::_cNBytes >= sizeof(void*), "Need to store a next pointer for a linked list of chunks.");

private: // variables
  std::atomic<void*> *_memChunks;
  std::atomic<size_t> _totalUnits;
  std::atomic<size_t> _maxTotalUnits;

private: // methods
  void FreeChunk(const size_t iSlot) {
    void *p = _memChunks[iSlot].load(std::memory_order_relaxed);
    while (p != nullptr) {
      void *next = *SRCast::CPtr<void*>(p); // note void* template argument here - that's to receive void**
      _mm_free(p);
      p = next;
    }
  }

public:
  static_assert((taNGranules * sizeof(std::atomic<void*>)) % SRSimd::_cNBytes == 0,
    "For SIMD efficiency, choose taNGranules divisable by larger power of 2.");

  explicit SRMemPool(const size_t maxTotalUnits = (512 * 1024 * 1024) / _cNUnitBytes)
    : _totalUnits(0), _maxTotalUnits(maxTotalUnits)
  {
    const size_t nMemChunksBytes = taNGranules * sizeof(*_memChunks);
    _memChunks = static_cast<decltype(_memChunks)>(_mm_malloc(nMemChunksBytes, SRSimd::_cNBytes));
    if (_memChunks == nullptr) {
      throw SRException(SRMessageBuilder(__FUNCTION__ " failed to allocate _memChunks of ")(nMemChunksBytes)(" bytes.")
        .GetOwnedSRString());
    }
    //TODO: vectorize/parallelize
    for (size_t i = 0; i < taNGranules; i++) {
      new(_memChunks + i) std::atomic<void*>(nullptr);
    }
  }

  virtual ~SRMemPool() override final {
    //TODO: vectorize/parallelize
    for (size_t i = 0; i < taNGranules; i++) {
      FreeChunk(i);
      _memChunks[i].~atomic<void*>();
    }
    _mm_free(_memChunks);
  }

  // This method is not thread-safe.
  virtual void FreeAllChunks() override final {
    std::atomic_thread_fence(std::memory_order_acquire);
    for (size_t i = 0; i < taNGranules; i++) {
      FreeChunk(i);
      _memChunks[i].store(nullptr, std::memory_order_relaxed);
    }
    _totalUnits.store(0, std::memory_order_release);
  }

  SRMemPool(const SRMemPool&) = delete;
  SRMemPool& operator=(const SRMemPool&) = delete;
  SRMemPool(SRMemPool&&) = delete;
  SRMemPool& operator=(SRMemPool&&) = delete;

  ATTR_RESTRICT virtual void* AllocMem(const size_t nBytes) override final {
    const size_t iSlot = (nBytes + _cNUnitBytes - 1) >> _cLogNUnitBytes;
    if (iSlot >= taNGranules) {
      return _mm_malloc(iSlot * _cNUnitBytes, _cNUnitBytes);
    }
    if (iSlot <= 0) {
      return nullptr;
    }
    std::atomic<void*>& head = _memChunks[iSlot];
    void *next;
    void* expected = head.load(std::memory_order_acquire);
    do {
      if (expected == nullptr) {
        _totalUnits.fetch_add(iSlot, std::memory_order_relaxed);
        return _mm_malloc(iSlot * _cNUnitBytes, _cNUnitBytes);
      }
      next = *SRCast::CPtr<void*>(expected); // note void* template argument here - that's to receive void**
    } while (!head.compare_exchange_weak(expected, next, std::memory_order_acq_rel, std::memory_order_acquire));
    return expected;
  }

  void ReleaseMem(void *PTR_RESTRICT p, const size_t nBytes) override final {
    const size_t iSlot = (nBytes + _cNUnitBytes - 1) >> _cLogNUnitBytes;
    if (iSlot >= taNGranules || iSlot <= 0) {
      _mm_free(p);
      return;
    }
    if (p == nullptr) {
      return;
    }
    if (_totalUnits.load(std::memory_order_relaxed) > _maxTotalUnits.load(std::memory_order_relaxed)) {
      _totalUnits.fetch_sub(iSlot, std::memory_order_relaxed);
      _mm_free(p);
      return;
    }
    std::atomic<void*>& PTR_RESTRICT head = _memChunks[iSlot];
    void *PTR_RESTRICT expected = head.load(std::memory_order_acquire);
    do {
      *SRCast::Ptr<void*>(p) = expected;
    } while (!head.compare_exchange_weak(expected, p, std::memory_order_release, std::memory_order_relaxed));
  }

  void SetMaxTotalUnits(const size_t nUnits) {
    _maxTotalUnits.store(nUnits, std::memory_order_relaxed);
  }
};

// MPP - memory pool pointer
template <typename taItem> class SRSmartMPP {
  SRBaseMemPool *_pMp;
  taItem *_pItems;
  size_t _nBytes;
public:
  explicit SRSmartMPP(SRBaseMemPool &mp, const size_t nItems) : _pMp(&mp), _nBytes(nItems * sizeof(taItem)) {
    _pItems = static_cast<taItem*>(_pMp->AllocMem(_nBytes));
    if (_pItems == nullptr) {
      throw SRException(SRMessageBuilder(SR_FILE_LINE " Failed to allocate ")(nItems)(" items, ")(sizeof(taItem))
        (" bytes each on memory pool ")(intptr_t(_pMp)).GetOwnedSRString());
    }
  }

  taItem* Get() const {
    return _pItems;
  }

  taItem* Detach() {
    taItem* answer = _pItems;
    _pItems = nullptr;
    return answer;
  }

  void EarlyRelease() {
    _pMp->ReleaseMem(_pItems, _nBytes);
    _pItems = nullptr;
  }

  ~SRSmartMPP() {
    _pMp->ReleaseMem(_pItems, _nBytes);
  }
};

template<typename taClass> void SRCheckingRelease(SRBaseMemPool& mp, taClass *pObj) {
  if (pObj != nullptr) {
    pObj->~taClass();
    mp.ReleaseMem(pObj, sizeof(taClass));
  }
}

template<typename taClass> class SRObjectMPP {
  SRBaseMemPool *_pMp;
  taClass *_pObj;

public:
  template<typename ...Args> explicit SRObjectMPP(SRBaseMemPool &mp, Args && ...constructorArgs) : _pMp(&mp) {
    _pObj = static_cast<taClass*>(_pMp->AllocMem(sizeof(taClass)));
    if (_pObj == nullptr) {
      throw SRException(SRMessageBuilder(SR_FILE_LINE " Failed to allocate ")(typeid(taClass).name())(" object on"
        " memory pool ")(intptr_t(_pMp)).GetOwnedSRString());
    }
    new(static_cast<void*>(_pObj)) taClass(std::forward<Args>(constructorArgs)...);
  }
  ~SRObjectMPP() {
    SRCheckingRelease(*_pMp, _pObj);
  }
  taClass* Get() const {
    return _pObj;
  }
  taClass* Detach() {
    taClass *answer = _pObj;
    _pObj = 0;
    return answer;
  }
  // If taClass destructor throws, we don't call it again from smart pointer destructor.
  void EarlyRelease() {
    taClass *p = _pObj;
    _pObj = nullptr;
    SRCheckingRelease(*_pMp, p);
  }
};

struct SRMemTotal {
  size_t _nBytes = 0;
};

enum class SRMemPadding : uint8_t {
  None = 0,
  Left = 1,
  Right = 2,
  Both = 3
};

struct SRMemItem {
  size_t _offs;
  ATTR_NOALIAS SRMemItem(const size_t nBytes, const SRMemPadding pad, SRMemTotal &PTR_RESTRICT mt) {
    const uint8_t u8pad = static_cast<uint8_t>(pad);
    _offs = ((u8pad & uint8_t(SRMemPadding::Left)) ? SRSimd::GetPaddedBytes(mt._nBytes) : mt._nBytes);
    const size_t newTotal = _offs + nBytes;
    mt._nBytes = ((u8pad & uint8_t(SRMemPadding::Right)) ? SRSimd::GetPaddedBytes(newTotal) : newTotal);
  }
};

template<typename taValue, typename taMemPool> struct SRCompressedMP;

template<typename taValue> struct SRCompressedMP<taValue, SRBaseMemPool> {
  taValue _value;

public: // methods
  SRBaseMemPool& GetMemPool() { return SRGetBaseMemPool(); }
};

template<typename taValue, typename taMemPool> struct SRCompressedMP {
  taValue _value;
  taMemPool *_pMp;

public: // methods
  taMemPool& GetMemPool() { return *_pMp; }
};

} // namespace SRPlat
