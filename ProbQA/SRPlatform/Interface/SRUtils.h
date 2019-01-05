// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRString.h"
#include "../SRPlatform/Interface/SRException.h"
#include "../SRPlatform/Interface/SRMessageBuilder.h"
#include "../SRPlatform/Interface/SRExitCode.h"
#include "../SRPlatform/Interface/SRCpuInfo.h"
#include "../SRPlatform/Interface/SRSimd.h"
#include "../SRPlatform/Interface/SRMacros.h"

namespace SRPlat {

extern "C" {

// pFirstCl and pLimCl must be cache-line-aligned pointers, so that the distance between them is a multiple of clSize .
// clSize is the CPU cache line size passed here to simplify assembly code.
// This function always flushes pFirstCl, then checks whether the address (plus cache line size) exceeds pLimCl .
SRPLATFORM_API ATTR_NOALIAS void __fastcall SRFlushCache(const void *PTR_RESTRICT pFirstCl,
  const void *PTR_RESTRICT pLimCl, const size_t clSize);

}

class SRPLATFORM_API SRUtils {
public: // Methods
  ATTR_NORETURN static void ExitProgram(SRExitCode code);

  static void RequestDebug();

  static SRString PrintUtcTimestamp();
  static SRString PrintUtcDate();
  template<bool taSubmillisecond> SRPLATFORM_API static SRString PrintUtcTime();

  template<bool taCache> SRPLATFORM_API ATTR_NOALIAS static void FillZeroVects(__m256i *PTR_RESTRICT p,
    const size_t nVects);
  
  // Copy nVects of 256-bit vectors.
  template<bool taCacheStore, bool taCacheLoad> ATTR_NOALIAS inline static void Copy256(
    void *const PTR_RESTRICT pStore, const void *const PTR_RESTRICT pLoad, size_t nVects);

  // Both store and load pointers are unaligned
  template<bool taCacheStore, bool taCacheLoad> SRPLATFORM_API ATTR_NOALIAS inline static void CopyUnalign(
    void *const PTR_RESTRICT pStore, const void *const PTR_RESTRICT pLoad, const size_t nBytes);
  
  // Store Aligned, Load Unaligned
  template<bool taCacheStore, bool taCacheLoad> SRPLATFORM_API ATTR_NOALIAS inline static void CopySaLu(
    void *const PTR_RESTRICT pStore, const void *const PTR_RESTRICT pLoad, const size_t nBytes);

  // Store Unaligned, Load Aligned
  template<bool taCacheStore, bool taCacheLoad> SRPLATFORM_API ATTR_NOALIAS inline static void CopySuLa(
    void *const PTR_RESTRICT pStore, const void *const PTR_RESTRICT pLoad, const size_t nBytes);

  // Broadcast item to all components of 256-bit vector register.
  template<typename taItem> ATTR_NOALIAS inline static
    typename std::enable_if_t<sizeof(__m256i) % sizeof(taItem) == 0, __m256i> __vectorcall Set1(const taItem item);

  //NOTE: it does nothing if p is perfectly aligned for SIMD.
  template<size_t taGran> ATTR_NOALIAS inline static void* __vectorcall FillPrologue(void *PTR_RESTRICT p,
    const __m256i vect);
  // pLim must point just behind the end of the region to fill.
  //NOTE: it does nothing if pLim is perfectly aligned for SIMD.
  template<size_t taGran> ATTR_NOALIAS inline static void* __vectorcall FillEpilogue(void *PTR_RESTRICT pLim,
    const __m256i vect);

  // taFlushLeft and taFlushRight indicate whether the partial cache lines in the beginning and in the end of the array
  //   should be flushed too.
  //NOTE: a fence is needed after a (series of) call(s) to this function.
  template<bool taFlushLeft, bool taFlushRight> ATTR_NOALIAS inline static void FlushCache(
    const void *PTR_RESTRICT pStart, const size_t nBytes);

  // paddedBytes must be a multiple of SIMD size
  ATTR_RESTRICT inline static void* ThrowingSimdAlloc(const size_t paddedBytes);
};

template<bool taCacheStore, bool taCacheLoad> ATTR_NOALIAS inline
void SRUtils::Copy256(void *const PTR_RESTRICT pStore, const void *const PTR_RESTRICT pLoad, size_t nVects)
{
  const __m256i *PTR_RESTRICT pSrc = SRCast::CPtr<__m256i>(pLoad);
  __m256i *PTR_RESTRICT pDest = SRCast::Ptr<__m256i>(pStore);
  for (; nVects > 0; nVects--, pSrc++, pDest++) {
    const __m256i loaded = taCacheLoad ? _mm256_load_si256(pSrc) : _mm256_stream_load_si256(pSrc);
    taCacheStore ? _mm256_store_si256(pDest, loaded) : _mm256_stream_si256(pDest, loaded);
  }
  if (!taCacheStore) {
    _mm_sfence();
  }
}

// Both store and load pointers are unaligned
template<bool taCacheStore, bool taCacheLoad> SRPLATFORM_API ATTR_NOALIAS inline
void SRUtils::CopyUnalign(void *const PTR_RESTRICT pStore, const void *const PTR_RESTRICT pLoad, const size_t nBytes) {
  //TODO: implement with AVX2 (stream) store/load 32 bytes at once
  memcpy(pStore, pLoad, nBytes);
}

// Store Aligned, Load Unaligned
template<bool taCacheStore, bool taCacheLoad> SRPLATFORM_API ATTR_NOALIAS inline
void SRUtils::CopySaLu(void *const PTR_RESTRICT pStore, const void *const PTR_RESTRICT pLoad, const size_t nBytes) {
  //TODO: implement with AVX2 (stream) store/load 32 bytes at once
  memcpy(pStore, pLoad, nBytes);
}

// Store Unaligned, Load Aligned
template<bool taCacheStore, bool taCacheLoad> SRPLATFORM_API ATTR_NOALIAS inline
void SRUtils::CopySuLa(void *const PTR_RESTRICT pStore, const void *const PTR_RESTRICT pLoad, const size_t nBytes) {
  //TODO: implement with AVX2 (stream) store/load 32 bytes at once
  memcpy(pStore, pLoad, nBytes);
}

template<typename taItem> ATTR_NOALIAS inline typename std::enable_if_t<sizeof(__m256i) % sizeof(taItem) == 0, __m256i>
__vectorcall SRUtils::Set1(const taItem item)
{
  switch (sizeof(item)) {
  case 1:
    return _mm256_set1_epi8(SRCast::Bitwise<int8_t>(item));
  case 2:
    return _mm256_set1_epi16(SRCast::Bitwise<int16_t>(item));
  case 4:
    return _mm256_set1_epi32(SRCast::Bitwise<int32_t>(item));
  case 8:
    return _mm256_set1_epi64x(SRCast::Bitwise<int64_t>(item));
  case 16:
    return _mm256_broadcastsi128_si256(SRCast::Bitwise<__m128i>(item));
  case 32:
    return SRCast::Bitwise<__m256i>(item);
  default:
    throw SRException(SRMessageBuilder("Unreachable ")(sizeof(taItem)).GetOwnedSRString());
  }
}

template<size_t taGran> ATTR_NOALIAS inline void* __vectorcall SRUtils::FillPrologue(void *PTR_RESTRICT p,
  const __m256i vect)
{
  static_assert(sizeof(vect) % taGran == 0, "Wrong granularity: there must be integer number of granules in SIMD.");
  switch (taGran) {
  default:
    SR_UNREACHABLE;
  case 1:
    if (uintptr_t(p) & 1) {
      uint8_t* pSpec = SRCast::Ptr<uint8_t>(p);
      *pSpec = vect.m256i_u8[0];
      p = pSpec + 1;
    }
    // fall through
  case 2:
    if (uintptr_t(p) & 2) {
      uint16_t* pSpec = SRCast::Ptr<uint16_t>(p);
      //TODO: check if it's faster than _mm_storeu_si16(pSpec, _mm256_castsi256_si128(vect))
      *pSpec = vect.m256i_u16[0];
      p = pSpec + 1;
    }
    // fall through
  case 4:
    if (uintptr_t(p) & 4) {
      uint32_t* pSpec = SRCast::Ptr<uint32_t>(p);
      //TODO: check if it's faster than _mm_storeu_si32()
      *pSpec = vect.m256i_u32[0];
      p = pSpec + 1;
    }
    // fall through
  case 8:
    if (uintptr_t(p) & 8) {
      uint64_t* pSpec = SRCast::Ptr<uint64_t>(p);
      //TODO: check if it's faster than _mm_storeu_si64() or _mm_storel_pi/_mm_storel_pd
      *pSpec = vect.m256i_u64[0];
      p = pSpec + 1;
    }
    // fall through
  case 16:
    if (uintptr_t(p) & 16) {
      __m128i* pSpec = SRCast::Ptr<__m128i>(p);
      _mm_store_si128(pSpec, _mm256_castsi256_si128(vect));
      p = pSpec + 1;
    }
  }
  return p;
}

template<size_t taGran> ATTR_NOALIAS inline void* __vectorcall SRUtils::FillEpilogue(void *PTR_RESTRICT pLim,
  const __m256i vect)
{
  static_assert(sizeof(vect) % taGran == 0, "Wrong granularity: there must be integer number of granules in SIMD.");
  switch (taGran) {
  default:
    SR_UNREACHABLE;
  case 1:
    if (uintptr_t(pLim) & 1) {
      uint8_t* pSpec = SRCast::Ptr<uint8_t>(pLim) - 1;
      *pSpec = vect.m256i_u8[0];
      pLim = pSpec;
    }
    //fall through
  case 2:
    if (uintptr_t(pLim) & 2) {
      uint16_t* pSpec = SRCast::Ptr<uint16_t>(pLim) - 1;
      *pSpec = vect.m256i_u16[0];
      pLim = pSpec;
    }
    //fall through
  case 4:
    if (uintptr_t(pLim) & 4) {
      uint32_t* pSpec = SRCast::Ptr<uint32_t>(pLim) - 1;
      *pSpec = vect.m256i_u32[0];
      pLim = pSpec;
    }
    //fall through
  case 8:
    if (uintptr_t(pLim) & 8) {
      uint64_t* pSpec = SRCast::Ptr<uint64_t>(pLim) - 1;
      *pSpec = vect.m256i_u64[0];
      pLim = pSpec;
    }
    //fall through
  case 16:
    if (uintptr_t(pLim) & 16) {
      __m128i* pSpec = SRCast::Ptr<__m128i>(pLim) - 1;
      _mm_store_si128(pSpec, _mm256_castsi256_si128(vect));
      pLim = pSpec;
    }
  }
  return pLim;
}

template<bool taFlushLeft, bool taFlushRight> ATTR_NOALIAS inline void SRUtils::FlushCache(
  const void *PTR_RESTRICT pStart, const size_t nBytes)
{
  uintptr_t addrStart = reinterpret_cast<uintptr_t>(pStart);
  uintptr_t addrLim = addrStart + nBytes;
  const uintptr_t clStart = addrStart & ~SRCpuInfo::_cacheLineMask;
  const uintptr_t clLim = addrLim & ~SRCpuInfo::_cacheLineMask;
  if (clStart == clLim) {
    // The whole array fits one cache line
#pragma warning( push )
#pragma warning( disable : 4127 ) // conditional expression is constant
    if (taFlushRight && (taFlushLeft || (addrStart == clStart))) {
#pragma warning( pop )
      //TODO: _mm_clflushopt() should be faster when available
      _mm_clflush(reinterpret_cast<const void*>(clStart));
    }
    return;
  }
  assert(clStart <= addrStart);
  if (clStart != addrStart) {
    if (taFlushLeft) {
      addrStart = clStart;
    } else {
      addrStart = clStart + SRCpuInfo::_cacheLineBytes;
    }
  }
  assert(clLim <= addrLim);
  if (clLim != addrLim) {
    if (taFlushRight) {
      addrLim = clLim + SRCpuInfo::_cacheLineBytes;
    } else {
      addrLim = clLim;
    }
  }
  if (addrStart < addrLim) {
    SRFlushCache(reinterpret_cast<const void*>(addrStart), reinterpret_cast<const void*>(addrLim),
      SRCpuInfo::_cacheLineBytes);
  }
}

ATTR_RESTRICT inline void* SRUtils::ThrowingSimdAlloc(const size_t paddedBytes) {
  assert(paddedBytes % SRSimd::_cNBytes == 0);
  void *PTR_RESTRICT ans = _mm_malloc(paddedBytes, SRSimd::_cNBytes);
  if (ans == nullptr) {
    throw SRException(SRMessageBuilder(SR_FILE_LINE " failed to allocate ")(paddedBytes)(" bytes.").GetOwnedSRString());
  }
  return ans;
}

} // namespace SRPlat
