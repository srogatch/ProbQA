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

template<> SRString SRUtils::PrintUtcTime<true>() {
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

template<> SRString SRUtils::PrintUtcTime<false>() {
  SYSTEMTIME st;
  GetSystemTime(&st);
  char buffer[16];
  snprintf(buffer, sizeof(buffer), "%.2hu:%.2hu:%.2hu.%.3hu", (unsigned short)st.wHour, (unsigned short)st.wMinute,
    (unsigned short)st.wSecond, (unsigned short)st.wMilliseconds);
  return SRString::MakeClone(buffer);
}

} // namespace SRPlat
