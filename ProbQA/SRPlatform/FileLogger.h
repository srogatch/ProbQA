#pragma once

#include "../SRPlatform/Interface/ISRLogger.h"
#include "../SRPlatform/Interface/SRCriticalSection.h"
#include "../SRPlatform/Interface/SRConditionVariable.h"

namespace SRPlat {

class FileLogger : public ISRLogger {
public: // constants
  uint32_t cMaxEnqueuedLen = 1024 * 1024;
private: // variables
  SRCriticalSection _cs;
  std::queue<std::string> _qu;
  SRConditionVariable _canPush;
  SRConditionVariable _canPop;
  std::thread _writerThread;
  FILE *_fpout;
  int64_t _enqueuedLen = 0; // total length of all strings enqueued
public:
  explicit FileLogger(const SRString& baseName);
  virtual ~FileLogger() override;
  virtual bool Log(const Severity s, const SRString& message) override;
};

} // namespace SRPlat
