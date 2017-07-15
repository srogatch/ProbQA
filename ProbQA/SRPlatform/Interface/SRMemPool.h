// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRException.h"
#include "../SRPlatform/Interface/SRSimd.h"

namespace SRPlat {

// If the unit is 256-bit, then taLogUnitBits should be 8 because 256 == (1<<8) .
// This class is thread-safe, except some methods explicitly specified as not thread-safe.
template<uint32_t taLogNUnitBits, uint32_t taNGranules> class SRMemPool {
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
      void *next = *reinterpret_cast<void**>(p);
      _mm_free(p);
      p = next;
    }
  }

public:
  static_assert((taNGranules * sizeof(std::atomic<void*>)) % SRSimd::_cNBytes == 0,
    "For SIMD efficiency, choose taNGranules divisable by larger power of 2.");

  explicit SRMemPool(const size_t maxTotalUnits = (512 * 1024 * 1024) / _cUnitBytes)
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

  ~SRMemPool() {
    //TODO: vectorize/parallelize
    for (size_t i = 0; i < taNGranules; i++) {
      FreeChunk(i);
      _memChunks[i].~atomic<void*>();
    }
    _mm_free(_memChunks);
  }

  // This method is not thread-safe.
  void FreeAllChunks() {
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

  void* AllocMem(const size_t nBytes) {
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
      next = *reinterpret_cast<void**>(expected);
    } while (!head.compare_exchange_weak(expected, next, std::memory_order_acq_rel, std::memory_order_acquire));
    return expected;
  }

  void ReleaseMem(void *p, const size_t nBytes) {
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
    std::atomic<void*>& head = _memChunks[iSlot];
    void *expected = head.load(std::memory_order_acquire);
    do {
      *reinterpret_cast<void**>(p) = expected;
    } while (!head.compare_exchange_weak(expected, p, std::memory_order_release, std::memory_order_relaxed));
  }

  void SetMaxTotalUnits(const size_t nUnits) {
    _maxTotalUnits.store(nUnits, std::memory_order_relaxed);
  }
};

// MPP - memory pool pointer
template <typename taMemPool, typename taItem> class SRSmartMPP {
  taMemPool *_pMp;
  taItem *_pItem;
  size_t _nBytes;
public:
  explicit SRSmartMPP(taMemPool &mp, const size_t nItems) : _pMp(&mp), _nBytes(nItems * sizeof(taItem)) {
    _pItem = static_cast<taItem*>(_pMp->AllocMem(_nBytes));
    if (_pItem == nullptr) {
      throw SRException(SRMessageBuilder(__FUNCTION__ " Failed to allocate ")(nItems)(" items, ")(sizeof(taItem))
        (" bytes each.").GetOwnedSRString());
    }
  }

  taItem* Get() {
    return _pItem;
  }

  taItem* Detach() {
    taItem* answer = _pItem;
    _pItem = nullptr;
    return answer;
  }

  void EarlyRelease() {
    _pMp->ReleaseMem(_pItem, _nBytes);
    _pItem = nullptr;
  }

  ~SRSmartMPP() {
    _pMp->ReleaseMem(_pItem, _nBytes);
  }
};

} // namespace SRPlat
