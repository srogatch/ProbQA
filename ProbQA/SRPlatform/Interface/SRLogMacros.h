// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRDefaultLogger.h"


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
