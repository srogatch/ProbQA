#include "stdafx.h"
#include "../SRPlatform/Interface/SRUtils.h"

namespace SRPlat {

SRString SRUtils::PrintTimestamp() {
  SYSTEMTIME st;
  GetLocalTime(&st);
  char buffer[128];
  snprintf(buffer, sizeof(buffer), "%d-%.2d-%.2d %.2d:%.2d:%.2d.%.3d", st.wYear, st.wMonth,
    st.wDay, st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
  return SRString::MakeClone(buffer);
}

} // namespace SRPlat
