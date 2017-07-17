// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once
#include "../SRPlatform/Interface/SRException.h"
#include "../SRPlatform/Interface/SRMessageBuilder.h"

namespace SRPlat {

class SRPLATFORM_API SRCannotOpenLogFileException : public SRException {
  SRString _fileName;
public:
  explicit SRCannotOpenLogFileException(const std::string& fileName) : _fileName(fileName),
    SRException(SRString::MakeUnowned("Cannot open log file.")) { }

  SRCannotOpenLogFileException(const SRCannotOpenLogFileException &fellow)
    : _fileName(fellow._fileName), SRException(fellow) { }

  SRCannotOpenLogFileException(SRCannotOpenLogFileException &&fellow)
    : _fileName(std::forward<SRString>(fellow._fileName)), SRException(std::forward<SRException>(fellow)) { }

  SREXCEPTION_TYPICAL(SRCannotOpenLogFile);

  virtual SRString ToString() override {
    return SRMessageBuilder()(GetMsg())('[')(_fileName)(']').GetOwnedSRString();
  }
};

class SRPLATFORM_API SRLoggerShutDownException : public SRException {
  SRString _unloggedMsg;
public:
  explicit SRLoggerShutDownException(const std::string& unloggedMsg) : _unloggedMsg(unloggedMsg),
    SRException(SRString::MakeUnowned("Cannot log because the logger is shut(ting) down.")) { }

  SRLoggerShutDownException(const SRLoggerShutDownException &fellow)
    : _unloggedMsg(fellow._unloggedMsg), SRException(fellow) { }

  SRLoggerShutDownException(SRLoggerShutDownException &&fellow)
    : _unloggedMsg(std::forward<SRString>(fellow._unloggedMsg)), SRException(std::forward<SRException>(fellow)) { }

  SREXCEPTION_TYPICAL(SRLoggerShutDown);

  virtual SRString ToString() override {
    return SRMessageBuilder()(GetMsg())('[')(_unloggedMsg)(']').GetOwnedSRString();
  }
};

} // namespace SRPlat
