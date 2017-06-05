#pragma once

#include "../SRPlatform/Interface/ISRLogger.h"
#include "../SRPlatform/Interface/SRCriticalSection.h"
#include "../SRPlatform/Interface/SRConditionVariable.h"

namespace SRPlat {

class FileLogger : public ISRLogger {
public: // constants
  uint32_t cMaxEnqueuedLen = 1024 * 1024;
private: // variables
  //// Cache-sensitive data
  SRCriticalSection _cs;
  std::queue<std::string> _qu;
  SRConditionVariable _canPush;
  SRConditionVariable _canPop;
  FILE *_fpout;
  uint64_t _enqueuedLen : 63; // total length of all strings enqueued
  uint64_t _bShutdown : 1;

  //// Cache-insensitive data
  std::thread _thrWriter;
  std::string _fileName;

  static const char* const cShutdownToken;

protected: // Methods
  void WriterEntry();
  void RequestShutdown();
  
public: // Methods
  explicit FileLogger(const SRString& baseName);
  virtual ~FileLogger() override;
  virtual bool Log(const Severity s, const SRString& message) override;
  virtual SRString GetFileName() override;
};

} // namespace SRPlat
