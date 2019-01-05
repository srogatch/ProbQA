// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRMemPool.h"

namespace SRPlat {

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

  // See https://stackoverflow.com/questions/54050890/how-to-utilize-template-copymove-constructor-and-assignment-operator
  SRMPAllocator(const SRMPAllocator& fellow) : _pMemPool(fellow._pMemPool) { }

  template <typename U> SRMPAllocator(const SRMPAllocator<U, taMemPool>& fellow) noexcept
    : _pMemPool(fellow._pMemPool) { }

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
