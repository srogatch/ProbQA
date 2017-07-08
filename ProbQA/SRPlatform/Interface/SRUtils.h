// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRString.h"

namespace SRPlat {

class SRPLATFORM_API SRUtils {
public: // Methods
  static SRString PrintUtcTimestamp();
  static SRString PrintUtcDate();
  template<bool taSubmillisecond> SRPLATFORM_API static SRString PrintUtcTime();
  template<bool taCache> SRPLATFORM_API static void FillZeroVects(__m256i *p, const size_t nVects);
  
  // Copy nVects of 256-bit vectors.
  template<bool taCacheStore, bool taCacheLoad> SRPLATFORM_API inline static void Copy256(void *pStore,
    const void *pLoad, size_t nVects);
  // Broadcast item to all components of 256-bit vector register.
  template<typename taItem> inline static enable_if_t<sizeof(__m256i) % sizeof(taItem) == 0, __m256i>
  Set1(const taItem& item);
  //NOTE: it doesn't call _mm_sfence() . Let the client decide about a better time to call it.
  //NOTE: it does nothing if p is perfectly aligned for SIMD.
  template<size_t taGran> inline static void* CopyPrologue(void *p, const __m256i vect);
};

template<bool taCacheStore, bool taCacheLoad> SRPLATFORM_API inline static 
void SRUtils::Copy256(void *pStore, const void *pLoad, size_t nVects)
{
  const __m256i *pSrc = reinterpret_cast<const __m256i*>(pLoad);
  __m256i *pDest = reinterpret_cast<__m256i*>(pStore);
  for (; nVects > 0; nVects--, pSrc++, pDest++) {
    const __m256i loaded = taCacheLoad ? _mm256_load_si256(pSrc) : _mm256_stream_load_si256(pSrc);
    taCacheStore ? _mm256_store_si256(pDest, loaded) : _mm256_stream_si256(pDest, loaded);
  }
  if (taCacheStore) {
    _mm_sfence();
  }
}

template<typename taItem> inline static enable_if_t<sizeof(__m256i) % sizeof(taItem) == 0, __m256i>
SRUtils::Set1(const taItem& item)
{
  switch (sizeof(item)) {
  case 1:
    return _mm256_set1_epi8(*reinterpret_cast<const int8_t*>(&item));
  case 2:
    return _mm256_set1_epi16(*reinterpret_cast<const int16_t*>(&item));
  case 4:
    return _mm256_set1_epi32(*reinterpret_cast<const int32_t*>(&item));
  case 8:
    return _mm256_set1_epi64(*reinterpret_cast<const int64_t*>(&item));
  case 16:
    return _mm256_broadcastsi128_si256(*reinterpret_cast<const __m128i*>(item));
  case 32:
    return *reinterpret_cast<const __m256i*>(&item);
  }
}

template<> inline static void* SRUtils::CopyPrologue<1>(void *p, const __m256i vect) {
  if (uintptr_t(p) & 1) {
    uint8_t* pSpec = reinterpret_cast<uint8_t*>(p);
    *pSpec = vect.m256i_u8[0];
    p = pSpec + 1;
  }
  return CopyPrologue<2>(p, vect); //TODO: check disassembly that this generates a tail call
}

template<> inline static void* SRUtils::CopyPrologue<2>(void *p, const __m256i vect) {
  if (uintptr_t(p) & 2) {
    uint16_t* pSpec = reinterpret_cast<uint16_t*>(p);
    *pSpec = vect.m256i_u16[0];
    p = pSpec + 1;
  }
  return CopyPrologue<4>(p, vect);
}

template<> inline static void* SRUtils::CopyPrologue<4>(void *p, const __m256i vect) {
  if (uintptr_t(p) & 4) {
    uint32_t* pSpec = reinterpret_cast<uint32_t*>(p);
    *pSpec = vect.m256i_u32[0];
    p = pSpec + 1;
  }
  return CopyPrologue<8>(p, vect);
}

template<> inline static void* SRUtils::CopyPrologue<8>(void *p, const __m256i vect) {
  if (uintptr_t(p) & 8) {
    uint64_t* pSpec = reinterpret_cast<uint64_t*>(p);
    *pSpec = vect.m256i_u64[0];
    p = pSpec + 1;
  }
  return CopyPrologue<16>(p, vect);
}

template<size_t taGran> inline static void* SRUtils::CopyPrologue(void *p, const __m256i vect) {
  static_assert(sizeof(vect) % taGran == 0, "Wrong granularity: there must be integer number of granules in SIMD.");
  switch (taGran) {
  case 1:
    if (uintptr_t(p) & 1) {
      uint8_t* pSpec = reinterpret_cast<uint8_t*>(p);
      *pSpec = vect.m256i_u8[0];
      p = pSpec + 1;
    }
    // fall through
  case 2:
    if (uintptr_t(p) & 2) {
      uint16_t* pSpec = reinterpret_cast<uint16_t*>(p);
      *pSpec = vect.m256i_u16[0];
      p = pSpec + 1;
    }
    // fall through
  case 4:
    if (uintptr_t(p) & 4) {
      uint32_t* pSpec = reinterpret_cast<uint32_t*>(p);
      *pSpec = vect.m256i_u32[0];
      p = pSpec + 1;
    }
    // fall through
  case 8:
    if (uintptr_t(p) & 8) {
      uint64_t* pSpec = reinterpret_cast<uint64_t*>(p);
      *pSpec = vect.m256i_u64[0];
      p = pSpec + 1;
    }
    // fall through
  case 16:
    if (uintptr_t(p) & 16) {
      __m128i* pSpec = reinterpret_cast<__m128i*>(p);
      _mm_store_si128(pSpec, _mm256_castsi256_si128(vect));
      p = pSpec + 1;
    }
  }
  return p;
}

} // namespace SRPlat
