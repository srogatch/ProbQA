// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../SRPlatform/DbgLogger.h"
#include "../SRPlatform/Interface/SRUtils.h"

namespace SRPlat {

bool DbgLogger::Log(const Severity sev, const SRString& message) {
  const std::string fullLine = SRUtils::PrintUtcTimestamp().ToString() + ' ' + std::to_string(sev) + ": "
    + message.ToString() + "\n";
  
  OutputDebugStringA(fullLine.c_str());
  return true; // Actually we don't know its fate because the above WinAPI call is void
}

SRString DbgLogger::GetFileName() {
  return SRString::MakeUnowned("Debugger Output Window");
}

} // namespace SRPlat
