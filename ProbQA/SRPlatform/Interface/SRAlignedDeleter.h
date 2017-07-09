#pragma once

namespace SRPlat {

class SRAlignedDeleter {
public:
  void operator()(void *p) { _mm_free(p); }
};

template<typename T> using AlignedUniquePtr = std::unique_ptr<T, SRAlignedDeleter>;

} // namespace SRPlat
