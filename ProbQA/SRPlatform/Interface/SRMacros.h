// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRDefaultLogger.h"

// Disable "conditional expression is constant" warning in `do { } while(false);` loops
#define WHILE_FALSE               \
  __pragma(warning(push))         \
  __pragma(warning(disable:4127)) \
  while(false)                    \
  __pragma(warning(pop))

#define SR_STRINGIZE(x) SR_STRINGIZE2(x)
#define SR_STRINGIZE2(x) #x
#define SR_LINE_STRING SR_STRINGIZE(__LINE__)

#define SR_COMBINE2(x, y) x ## y
#define SR_COMBINE(x, y) SR_COMBINE2(x, y)

#define SR_FILE_LINE __FILE__ "(" SR_LINE_STRING "):"

#define SR_LOG_WINFAIL(severityVar, loggerVar, lastErrVar) do { \
  (loggerVar)->Log( \
    ISRLogger::Severity::severityVar, \
    SRPlat::SRString( \
      std::string("Failed WinAPI call at " SR_FILE_LINE " GetLastError=") + std::to_string(lastErrVar) \
    ) \
  ); \
} WHILE_FALSE

#define SR_LOG_WINFAIL_GLE(severityVar, loggerVar) SR_LOG_WINFAIL(severityVar, loggerVar, GetLastError())

#define SR_DLOG_WINFAIL_GLE(severityVar) SR_LOG_WINFAIL_GLE(severityVar, SRDefaultLogger::Get())
