// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../SRPlatform/Interface/SRUtils.h"

namespace SRPlat {

SRString SRUtils::PrintUtcTimestamp() {
  SYSTEMTIME st;
  GetSystemTime(&st);
  char buffer[32];
  snprintf(buffer, sizeof(buffer), "%hu-%.2hu-%.2hu %.2hu:%.2hu:%.2hu.%.3hu", (unsigned short)st.wYear,
    (unsigned short)st.wMonth, (unsigned short)st.wDay, (unsigned short)st.wHour, (unsigned short)st.wMinute,
    (unsigned short)st.wSecond, (unsigned short)st.wMilliseconds);
  return SRString::MakeClone(buffer);
}

SRString SRUtils::PrintUtcDate() {
  SYSTEMTIME st;
  GetSystemTime(&st);
  char buffer[16];
  snprintf(buffer, sizeof(buffer), "%hu-%.2hu-%.2hu", (unsigned short)st.wYear, (unsigned short)st.wMonth,
    (unsigned short)st.wDay);
  return SRString::MakeClone(buffer);
}

template<> SRPLATFORM_API SRString SRUtils::PrintUtcTime<true>() {
  FILETIME ft;
  GetSystemTimePreciseAsFileTime(&ft);
  SYSTEMTIME st;
  char buffer[32];
  if (FileTimeToSystemTime(&ft, &st)) {
    const uint16_t subMiS = static_cast<uint16_t>( (*reinterpret_cast<const uint64_t*>(&ft)) % 10000 );
    snprintf(buffer, sizeof(buffer), "%.2hu:%.2hu:%.2hu.%.3hu.%3hu.%hu", (unsigned short)st.wHour,
      (unsigned short)st.wMinute, (unsigned short)st.wSecond, (unsigned short)st.wMilliseconds,
      (unsigned short)(subMiS / 10), (unsigned short)(subMiS%10));
  }
  else {
    // Can't log: this method may be used by the log.
    // Can't throw: we may lose the error being logged then.
    snprintf(buffer, sizeof(buffer), "FTTST() WinErr %u", (unsigned int)GetLastError());
  }
  return SRString::MakeClone(buffer);
}

template<> SRPLATFORM_API SRString SRUtils::PrintUtcTime<false>() {
  SYSTEMTIME st;
  GetSystemTime(&st);
  char buffer[16];
  snprintf(buffer, sizeof(buffer), "%.2hu:%.2hu:%.2hu.%.3hu", (unsigned short)st.wHour, (unsigned short)st.wMinute,
    (unsigned short)st.wSecond, (unsigned short)st.wMilliseconds);
  return SRString::MakeClone(buffer);
}

template<> SRPLATFORM_API static void SRUtils::FillZeroVects<true>(__m256i *p, const size_t nVects) {
  const __m256i vZero = _mm256_setzero_si256();
  for (void *pEn = p + nVects; p < pEn; p++) {
    _mm256_stream_si256(p, vZero);
  }
  //TODO: should here be rather _mm_mfence() ? https://stackoverflow.com/questions/44864033/make-previous-memory-stores-visible-to-subsequent-memory-loads
  _mm_sfence();
}

template<> SRPLATFORM_API static void SRUtils::FillZeroVects<false>(__m256i *p, const size_t nVects) {
  const __m256i vZero = _mm256_setzero_si256();
  for (void *pEn = p + nVects; p < pEn; p++) {
    _mm256_store_si256(p, vZero);
  }
}

} // namespace SRPlat
