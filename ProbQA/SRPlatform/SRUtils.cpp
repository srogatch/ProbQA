// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "../SRPlatform/Interface/SRMacros.h"
#include "../SRPlatform/Interface/SRUtils.h"

namespace SRPlat {

ATTR_NORETURN void SRUtils::ExitProgram(SRExitCode code) {
  std::quick_exit(static_cast<int>(code));
}

void SRUtils::RequestDebug() {
  if (!IsDebuggerPresent()) {
    std::wstring message(L"Connect the debugger to process ");
    message += std::to_wstring(GetCurrentProcessId());
    message += L", then close this dialog. Otherwise the process may terminate.";
    MessageBox(nullptr, message.c_str(), L"SRPlatform has requested a Debug", 0);
  }
  __debugbreak();
}

SRString SRUtils::PrintUtcTimestamp() {
  SYSTEMTIME st;
  GetSystemTime(&st);
  char buffer[32];
  snprintf(buffer, sizeof(buffer), "%hu-%.2hu-%.2hu %.2hu:%.2hu:%.2hu.%.3hu", CASTF_HU(st.wYear), CASTF_HU(st.wMonth),
    CASTF_HU(st.wDay), CASTF_HU(st.wHour), CASTF_HU(st.wMinute), CASTF_HU(st.wSecond), CASTF_HU(st.wMilliseconds));
  return SRString::MakeClone(buffer);
}

SRString SRUtils::PrintUtcDate() {
  SYSTEMTIME st;
  GetSystemTime(&st);
  char buffer[16];
  snprintf(buffer, sizeof(buffer), "%hu-%.2hu-%.2hu", CASTF_HU(st.wYear), CASTF_HU(st.wMonth), CASTF_HU(st.wDay));
  return SRString::MakeClone(buffer);
}

template<> SRPLATFORM_API SRString SRUtils::PrintUtcTime<true>() {
  FILETIME ft;
  GetSystemTimePreciseAsFileTime(&ft);
  SYSTEMTIME st;
  char buffer[32];
  if (FileTimeToSystemTime(&ft, &st)) {
    const uint16_t subMiS = static_cast<uint16_t>( (SRCast::Bitwise<uint64_t>(ft)) % 10000 );
    snprintf(buffer, sizeof(buffer), "%.2hu:%.2hu:%.2hu.%.3hu.%.3hu.%hu", CASTF_HU(st.wHour), CASTF_HU(st.wMinute),
      CASTF_HU(st.wSecond), CASTF_HU(st.wMilliseconds), CASTF_HU(subMiS / 10), CASTF_HU(subMiS%10));
  }
  else {
    // Can't log: this method may be used by the log.
    // Can't throw: we may lose the error being logged then.
    snprintf(buffer, sizeof(buffer), "FTTST() WinErr %u", CASTF_DU(GetLastError()));
  }
  return SRString::MakeClone(buffer);
}

template<> SRPLATFORM_API SRString SRUtils::PrintUtcTime<false>() {
  SYSTEMTIME st;
  GetSystemTime(&st);
  char buffer[16];
  snprintf(buffer, sizeof(buffer), "%.2hu:%.2hu:%.2hu.%.3hu", CASTF_HU(st.wHour), CASTF_HU(st.wMinute),
    CASTF_HU(st.wSecond), CASTF_HU(st.wMilliseconds));
  return SRString::MakeClone(buffer);
}

template<> SRPLATFORM_API ATTR_NOALIAS void
SRUtils::FillZeroVects<false>(__m256i *PTR_RESTRICT p, const size_t nVects) {
  const __m256i vZero = _mm256_setzero_si256();
  for (__m256i *pEn = p + nVects; p < pEn; p++) {
    _mm256_stream_si256(p, vZero);
  }
  //TODO: should here be rather _mm_mfence() ? https://stackoverflow.com/questions/44864033/make-previous-memory-stores-visible-to-subsequent-memory-loads
  _mm_sfence();
}

template<> SRPLATFORM_API ATTR_NOALIAS void
SRUtils::FillZeroVects<true>(__m256i *PTR_RESTRICT p, const size_t nVects) {
  const __m256i vZero = _mm256_setzero_si256();
  for (__m256i *pEn = p + nVects; p < pEn; p++) {
    _mm256_store_si256(p, vZero);
  }
}

} // namespace SRPlat
