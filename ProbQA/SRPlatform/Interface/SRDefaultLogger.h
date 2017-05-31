#pragma once

#include "../SRPlatform/Interface/ISRLogger.h"
#include "../SRPlatform/Interface/SRString.h"

namespace SRPlat {

class SRPLATFORM_API SRDefaultLogger {
public:
  // Timestamp and ".log" extension are added to the base name. So the base name should contain the file path and file
  //   name prefix.
  static void Init(const SRString& baseName);
  // Returns a file logger if initialized. Otherwise returns a logger that outputs to VS debug window.
  static ISRLogger* Get();
  // Explicitly get the debug-window logger.
  static ISRLogger* Dbg();
};

} // namespace SRPlat
