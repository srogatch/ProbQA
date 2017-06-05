#include "stdafx.h"
#include "../SRPlatform/DbgLogger.h"
#include "../SRPlatform/Interface/SRUtils.h"

namespace SRPlat {

bool DbgLogger::Log(const Severity sev, const SRString& message) {
  std::string fullLine = SRUtils::PrintUtcTimestamp().ToString() + ' ' + std::to_string(sev) + ": " + message.ToString();
  OutputDebugStringA(fullLine.c_str());
  return true; // Actually we don't know its fate because the above WinAPI call is void
}

SRString DbgLogger::GetFileName() {
  return SRString::MakeUnowned("Debugger Output Window");
}

} // namespace SRPlat
