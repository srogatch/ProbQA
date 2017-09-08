// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

namespace SRPlat {

union SRPacked64 {
  double _f64;
  float _f32[2];
  uint64_t _u64;
  int64_t _i64;
  uint32_t _u32[2];
  int32_t _i32[2];
  uint16_t _u16[4];
  int16_t _i16[4];
  uint8_t _u8[8];
  int8_t _i8[8];

  SRPacked64() { }

  constexpr SRPacked64(const __m128i& vect, const uint8_t at) : _u64(vect.m128i_u64[at]) { }
  SRPacked64(const __m128d& vect, const uint8_t at) : SRPacked64(_mm_castpd_si128(vect), at) { }
  SRPacked64(const __m128& vect, const uint8_t at) : SRPacked64(_mm_castps_si128(vect), at) { }

  constexpr explicit SRPacked64(const uint64_t value) : _u64(value) { }

  constexpr static SRPacked64 Set1U32(const uint32_t value) {
    return SRPacked64((uint64_t(value) << 32) | value);
  }

  template<uint8_t taAt> constexpr static SRPacked64 __vectorcall SetToComp(const __m128i& vect) {
    return SRPacked64(_mm_extract_epi64(vect, taAt));
  }
  template<uint8_t taAt> constexpr static SRPacked64 __vectorcall SetToComp(const __m128d& vect) {
    static_assert(taAt <= 1, "There are 2 components, 64 bit in each.");
    Packed64 ans;
    taAt ? _mm_storeh_pd(&ans._f64, vect) : _mm_storel_pd(&ans._f64, vect);
    return ans;
  }
  template<uint8_t taAt> constexpr static SRPacked64 __vectorcall SetToComp(const __m128& vect) {
    return SetToComp<taAt>(_mm_castps_pd(vect));
  }
};

} // namespace SRPlat
