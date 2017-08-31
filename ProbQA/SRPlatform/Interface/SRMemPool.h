// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRPlatform.h"
#include "../SRPlatform/Interface/SRException.h"
#include "../SRPlatform/Interface/SRMessageBuilder.h"
#include "../SRPlatform/Interface/SRSimd.h"

namespace SRPlat {

class SRPLATFORM_API SRBaseMemPool {
public:
  virtual ~SRBaseMemPool() { }
  virtual void FreeAllChunks() { }
  virtual void* AllocMem(const size_t nBytes) { return _mm_malloc(SRSimd::GetPaddedBytes(nBytes), SRSimd::_cNBytes); }
  virtual void ReleaseMem(void *p, const size_t) { _mm_free(p); }
};

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
      void *next = *reinterpret_cast<void**>(p);
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

  virtual void* AllocMem(const size_t nBytes) override final {
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

  void ReleaseMem(void *p, const size_t nBytes) override final {
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
template <typename taItem> class SRSmartMPP {
  SRBaseMemPool *_pMp;
  taItem *_pItem;
  size_t _nBytes;
public:
  explicit SRSmartMPP(SRBaseMemPool &mp, const size_t nItems) : _pMp(&mp), _nBytes(nItems * sizeof(taItem)) {
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

template<typename T, typename taMemPool> class SRMPAllocator {
  taMemPool *_pMemPool;
public:
  static constexpr size_t _cAlignment = taMemPool::_cNUnitBytes;
  template<size_t taMPAlign, size_t taItemAlign, size_t taItemSize> struct TAlignChecker {
    static constexpr bool isGood = (taMPAlign % taItemAlign == 0);
      //&& ((taMPAlign % taItemSize == 0) || (taItemSize % taMPAlign == 0));
    static_assert(isGood, "Memory pool unit size is too small for allocator value type.");
  };
  static constexpr bool _cbAlignGood = TAlignChecker<_cAlignment, alignof(T), sizeof(T)>::isGood;

  //// The following will be the same for virtually all allocators.
  typedef T * pointer;
  typedef const T * const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef T value_type;
  typedef std::size_t size_type;
  typedef ptrdiff_t difference_type;

  typedef std::true_type propagate_on_container_move_assignment;
  typedef std::true_type propagate_on_container_copy_assignment;
  typedef std::true_type propagate_on_container_swap;

  taMemPool* GetMemPool() const { return _pMemPool; }

  T * address(T& r) const noexcept {
    return &r;
  }

  const T * address(const T& s) const noexcept {
    return &s;
  }

  std::size_t max_size() const noexcept {
    // The following has been carefully written to be independent of the definition of size_t and to avoid
    // signed/unsigned warnings.
    return (static_cast<std::size_t>(0) - static_cast<std::size_t>(1)) / sizeof(T);
  }

  //// The following must be the same for all allocators.
  template <typename U>
  struct rebind {
    typedef SRMPAllocator<U, taMemPool> other;
  };

  template <class U, class ...Args> void construct(U* p, Args&&... args) const {
    void * const pv = static_cast<void *>(p);
    ::new (pv) U(std::forward<Args>(args)...);
  }

  void destroy(T * const p) const {
    p->~T();
  }

  // Default constructor, copy constructor, rebinding constructor, and destructor.
  // Empty for stateless allocators.
  explicit SRMPAllocator(taMemPool &memPool) noexcept : _pMemPool(&memPool) { }

  template <typename U> SRMPAllocator(const SRMPAllocator<U, taMemPool>& fellow) noexcept
    : _pMemPool(fellow.GetMemPool()) { }

  ~SRMPAllocator() noexcept { }


  // The following will be different for each allocator.
  T * allocate(const std::size_t n) const {
    return static_cast<T*>(_pMemPool->AllocMem(n * sizeof(T)));
  }

  void deallocate(T * const p, const std::size_t n) const {
    _pMemPool->ReleaseMem(p, n * sizeof(T));
  }


  // The following will be the same for all allocators that ignore hints.
  template <typename U>
  T * allocate(const std::size_t n, const U * /* const hint */) const {
    return allocate(n);
  }


  // Allocators are not required to be assignable, so all allocators should have a private unimplemented assignment
  // operator. Note that this will trigger the off-by-default (enabled under /Wall) warning C4626  "assignment operator
  // could not be generated because a base class assignment operator is inaccessible" within the STL headers, but that
  // warning is useless.
private:
  SRMPAllocator& operator=(const SRMPAllocator&);
};

template <typename T, typename MPT, typename U, typename MPU> inline
bool operator== (const SRMPAllocator<T, MPT>& a, const SRMPAllocator<U, MPU>& b) noexcept {
  if (a._cAlignment != b._cAlignment) {
    return false;
  }
  return static_cast<void*>(a.GetMemPool()) == static_cast<void*>(b.GetMemPool());
}

template <typename T, typename MPT, typename U, typename MPU> inline
bool operator!= (const SRMPAllocator<T, MPT>& a, const SRMPAllocator<U, MPU>& b) noexcept {
  if (a._cAlignment != b._cAlignment) {
    return true;
  }
  return static_cast<void*>(a.GetMemPool()) != static_cast<void*>(b.GetMemPool());
}

} // namespace SRPlat
