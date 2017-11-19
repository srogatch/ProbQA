// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../SRPlatform/FileLogger.h"
#include "../SRPlatform/Interface/SRUtils.h"
#include "../SRPlatform/Interface/Exceptions/SRFileLoggerExceptions.h"
#include "../SRPlatform/Interface/SRLock.h"

namespace SRPlat {

#define FPRINTF_LOG_PREPARE_TIME \
  SRString sTimeMacro = SRUtils::PrintUtcTime<false>(); \
  const char *pdTimeMacro; \
  size_t lenTimeMacro = sTimeMacro.GetData(pdTimeMacro);

#define FPRINTF_LOG_PREFIX "%.*s [%s] "

#define FPRINTF_LOG_FIRST_PARAMS(severityVar) \
  static_cast<int>(lenTimeMacro), pdTimeMacro, std::to_string(ISRLogger::Severity::severityVar).c_str()

FileLogger::FileLogger(const SRString& baseName) : _bShutdown(0), _enqueuedLen(0)
{
  _fileName = baseName.ToStd() + SRUtils::PrintUtcDate().ToStd() + ".log";
  _fpout = fopen(_fileName.c_str(), "at");
  if (_fpout == nullptr) {
    throw SRCannotOpenLogFileException(_fileName);
  }
  if (setvbuf(_fpout, nullptr, _IOFBF, cFileBufferBytes) != 0) {
    FPRINTF_LOG_PREPARE_TIME
    fprintf(_fpout, FPRINTF_LOG_PREFIX "Failed to set file buffer to %" PRIu32 " bytes. Logging may be slow.\n",
      FPRINTF_LOG_FIRST_PARAMS(Error), cFileBufferBytes);
    fflush(_fpout);
  }
  _thrWriter = std::thread(&FileLogger::WriterEntry, this);
}

FileLogger::~FileLogger() {
  RequestShutdown();
  _thrWriter.join();
  fclose(_fpout);
}

bool FileLogger::Log(const Severity sev, const SRString& message) {
  SRString sTime = SRUtils::PrintUtcTime<true>();
  std::string fullMessage = sTime.ToStd() + " [" + std::to_string(sev) + "] " + message.ToStd();
  bool bNotifyPoppers;
  {
    SRLock<SRCriticalSection> csl(_cs);
    for (;;) {
      if (_bShutdown) {
        csl.EarlyRelease();
        throw SRLoggerShutDownException(fullMessage);
      }
      if (_enqueuedLen < cMaxEnqueuedLen) {
        break;
      }
      _canPush.Wait(_cs);
    }
    _enqueuedLen += fullMessage.size();
    _qu.push(std::move(fullMessage));
    bNotifyPoppers = (_qu.size() == 1);
  }
  if (bNotifyPoppers) {
    _canPop.WakeAll();
  }
  return true;
}

SRString FileLogger::GetFileName() {
  return SRString(_fileName);
}

void FileLogger::RequestShutdown() {
  bool bWake = false;
  {
    SRLock<SRCriticalSection> csl(_cs);
    if (!_bShutdown) {
      _bShutdown = 1;
      bWake = true;
    }
  }
  if (bWake) {
    _canPop.WakeAll();
  }
}

void FileLogger::WriterEntry() {
  for (;;) 
  {
    std::string token;
    bool bNotifyPushers;
    bool bFlush;
    {
      SRLock<SRCriticalSection> csl(_cs);
      while (_qu.size() == 0) {
        if (_bShutdown) {
          return;
        }
        _canPop.Wait(_cs);
      }
      token = std::move(_qu.front());
      _qu.pop();
      bFlush = (_qu.size() == 0);

      const bool bWasSaturated = (_enqueuedLen >= cMaxEnqueuedLen);
      if (token.size() > _enqueuedLen) {
        //TODO: shall this logging be moved out of critical section?
        FPRINTF_LOG_PREPARE_TIME
        fprintf(_fpout, FPRINTF_LOG_PREFIX "Enqueued length %" PRIu64 " is smaller than token length %" PRIu64 "."
          " Setting enqueued length to 0.\n", FPRINTF_LOG_FIRST_PARAMS(Critical),
          static_cast<uint64_t>(_enqueuedLen), static_cast<uint64_t>(token.size()));
        bFlush = true;

        _enqueuedLen = 0;
      }
      else {
        _enqueuedLen -= token.size();
      }

      bNotifyPushers = (bWasSaturated && _enqueuedLen < cMaxEnqueuedLen);
    }
    if (bNotifyPushers) {
      _canPush.WakeAll();
    }
    fprintf(_fpout, "%s\n", token.c_str());
    if (bFlush) {
      fflush(_fpout);
    }
  }
}

} // namespace SRPlat