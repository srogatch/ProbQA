// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once
#include "../SRPlatform/Interface/SRException.h"

namespace SRPlat {

class SRPLATFORM_API SRCannotOpenLogFileException : public SRException {
  SRString _fileName;
public:
  explicit SRCannotOpenLogFileException(const std::string& fileName) : _fileName(fileName),
    SRException(SRString::MakeUnowned("Cannot open log file.")) { }
};

class SRPLATFORM_API SRLoggerShutDownException : public SRException {
  SRString _unloggedMsg;
public:
  explicit SRLoggerShutDownException(const std::string& unloggedMsg) : _unloggedMsg(unloggedMsg),
    SRException(SRString::MakeUnowned("Cannot log because the logger is shut(ting) down .")) { }
};

} // namespace SRPlat
