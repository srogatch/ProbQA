#include "stdafx.h"
#include "../SRPlatform/DbgLogger.h"
#include "../SRPlatform/Interface/SRUtils.h"

namespace SRPlat {

bool DbgLogger::Log(const Severity s, const SRString& message) {
  std::string fullLine = SRUtils::PrintTimestamp().ToString() + ' ' + std::to_string(s) + ": " + message.ToString();
  OutputDebugStringA(fullLine.c_str());
  return true; // Actually we don't know its fate because the above WinAPI call is void
}

} // namespace SRPlat
