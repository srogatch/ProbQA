#pragma once

namespace SRPlat {

// Based on:
//   https://gist.github.com/donny-dont/1471329#file-aligned_allocator-cpp
//   https://blogs.msdn.microsoft.com/vcblog/2008/08/28/the-mallocator/
//   https://stackoverflow.com/questions/12942548/making-stdvector-allocate-aligned-memory : improved construct()?
// See also:
//   https://www.codeproject.com/Articles/4795/C-Standard-Allocator-An-Introduction-and-Implement
//   https://stackoverflow.com/questions/8456236/how-is-a-vectors-data-aligned
template <typename T, std::size_t Alignment>
class SRAlignedAllocator {
public:

  //// The following will be the same for virtually all allocators.
  typedef T * pointer;
  typedef const T * const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef T value_type;
  typedef std::size_t size_type;
  typedef ptrdiff_t difference_type;

  typedef std::true_type propagate_on_container_move_assignment;
  //TODO: how shall the below be?
  //typedef std::true_type propagate_on_container_copy_assignment;
  //typedef std::true_type propagate_on_container_swap;

  T * address(T& r) const noexcept {
    return &r;
  }

  const T * address(const T& s) const noexcept {
    return &s;
  }

  std::size_t max_size() const noexcept {
    // The following has been carefully written to be independent of
    // the definition of size_t and to avoid signed/unsigned warnings.
    return (static_cast<std::size_t>(0) - static_cast<std::size_t>(1)) / sizeof(T);
  }

  //// The following must be the same for all allocators.
  template <typename U>
  struct rebind {
    typedef SRAlignedAllocator<U, Alignment> other;
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
  SRAlignedAllocator() noexcept { }

  template <typename U> SRAlignedAllocator(const SRAlignedAllocator<U, Alignment>&) noexcept { }

  ~SRAlignedAllocator() noexcept { }


  // The following will be different for each allocator.
  T * allocate(const std::size_t n) const {
    // The return value of allocate(0) is unspecified.
    // Mallocator returns NULL in order to avoid depending
    // on malloc(0)'s implementation-defined behavior
    // (the implementation can define malloc(0) to return NULL,
    // in which case the bad_alloc check below would fire).
    // All allocators can return NULL in this case.
    if (n == 0) {
      return NULL;
    }

    // All allocators should contain an integer overflow check.
    // The Standardization Committee recommends that std::length_error
    // be thrown in the case of integer overflow.
    if (n > max_size()) {
      throw std::length_error("aligned_allocator<T>::allocate() - Integer overflow.");
    }

    // Mallocator wraps malloc().
    void * const pv = _mm_malloc(n * sizeof(T), Alignment);
    //C++11 not supported in MSVC++: void * const pv = aligned_alloc(Alignment, n * sizeof(T));

    // Allocators should throw std::bad_alloc in the case of memory allocation failure.
    if (pv == NULL) {
      throw std::bad_alloc();
    }

    return static_cast<T *>(pv);
  }

  void deallocate(T * const p, const std::size_t n) const {
    _mm_free(p);
    //C++11 not supported in MSVC++ as a pair to aligned_alloc(): free(p);
    (void)n;
  }


  // The following will be the same for all allocators that ignore hints.
  template <typename U>
  T * allocate(const std::size_t n, const U * /* const hint */) const {
    return allocate(n);
  }


  // Allocators are not required to be assignable, so
  // all allocators should have a private unimplemented
  // assignment operator. Note that this will trigger the
  // off-by-default (enabled under /Wall) warning C4626
  // "assignment operator could not be generated because a
  // base class assignment operator is inaccessible" within
  // the STL headers, but that warning is useless.
private:
  SRAlignedAllocator& operator=(const SRAlignedAllocator&);
};

template <typename T, std::size_t TAlign, typename U, std::size_t UAlign> inline
bool operator== (const SRAlignedAllocator<T, TAlign>&, const SRAlignedAllocator<U, UAlign>&) noexcept {
  return TAlign == UAlign;
}

template <typename T, std::size_t TAlign, typename U, std::size_t UAlign> inline
bool operator!= (const SRAlignedAllocator<T, TAlign>&, const SRAlignedAllocator<U, UAlign>&) noexcept {
  return TAlign != UAlign;
}

} // namespace SRPlat