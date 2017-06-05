#include "stdafx.h"
#include "../SRPlatform/FileLogger.h"
#include "../SRPlatform/Interface/SRUtils.h"
#include "../SRPlatform/Interface/Exceptions/SRFileLoggerExceptions.h"
#include "../SRPlatform/Interface/SRLock.h"

namespace SRPlat {

#define FPRINTF_LOG_VARS \
  SRString sTime = SRUtils::PrintUtcTime<false>(); \
  const char *pdTime; \
  size_t lenTime = sTime.GetData(pdTime);

#define FPRINTF_LOG_PREFIX "%.*s [%s] "

#define FPRINTF_LOG_FIRSTPARAMS(severity) \
  static_cast<int>(lenTime), pdTime, std::to_string(ISRLogger::Severity::severity).c_str()

FileLogger::FileLogger(const SRString& baseName) : _bShutdown(0), _enqueuedLen(0)
{
  _fileName = baseName.ToString() + SRUtils::PrintUtcDate().ToString() + ".log";
  _fpout = fopen(_fileName.c_str(), "at");
  if (_fpout == nullptr) {
    throw SRCannotOpenLogFileException(_fileName);
  }
  if (setvbuf(_fpout, nullptr, _IOFBF, cFileBufferBytes) != 0) {
    SRString sTime = SRUtils::PrintUtcTime<false>();
    const char *pdTime;
    size_t lenTime = sTime.GetData(pdTime);
    fprintf(_fpout, "%.*s [%s] Failed to set file buffer to %" PRIu32 " bytes. Logging may be slow.\n",
      static_cast<int>(lenTime), pdTime, std::to_string(ISRLogger::Severity::Error).c_str(), cFileBufferBytes);
    fflush(_fpout);
  }
  _thrWriter = std::thread(&FileLogger::WriterEntry, this);
}

FileLogger::~FileLogger() {
  RequestShutdown();
  _thrWriter.join();
  fclose(_fpout);
}

bool FileLogger::Log(const Severity s, const SRString& message) {
  //TODO: implement
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
        SRString sTime = SRUtils::PrintUtcTime<false>();
        const char *pdTime;
        size_t lenTime = sTime.GetData(pdTime);
        fprintf(_fpout, "%.*s [%s] Enqueued length %" PRIu64 " is smaller than token length %" PRIu64 ". Setting"
          " enqueued length to 0.\n", static_cast<int>(lenTime), pdTime,
          std::to_string(ISRLogger::Severity::Critical).c_str(),
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