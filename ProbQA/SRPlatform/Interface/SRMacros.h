#pragma once

#include "../SRPlatform/Interface/SRDefaultLogger.h"

// Disable "conditional expression is constant" warning in `do { } while(false);` loops
#define WHILE_FALSE \
  __pragma(warning(push))         \
  __pragma(warning(disable:4127)) \
  while(false)					\
  __pragma(warning(pop))

#define SR_STRINGIZE(x) SR_STRINGIZE2(x)
#define SR_STRINGIZE2(x) #x
#define SR_LINE_STRING SR_STRINGIZE(__LINE__)

#define SR_LOG_WINFAIL(severityVar, loggerVar, lastErrVar) do { \
  (loggerVar)->Log( \
    ISRLogger::Severity::severityVar, \
    SRPlat::SRString( \
      std::string("Failed WinAPI call at " __FILE__ "(" SR_LINE_STRING "): GetLastError=") \
        + std::to_string(lastErrVar) \
    ) \
  ); \
} WHILE_FALSE

#define SR_LOG_WINFAIL_GLE(severityVar, loggerVar) SR_LOG_WINFAIL(severityVar, loggerVar, GetLastError())

#define SR_DLOG_WINFAIL_GLE(severityVar) SR_LOG_WINFAIL_GLE(severityVar, SRDefaultLogger::Get())
