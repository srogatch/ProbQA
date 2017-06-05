#include "stdafx.h"
#include "../SRPlatform/FileLogger.h"
#include "../SRPlatform/Interface/SRUtils.h"
#include "../SRPlatform/Interface/Exceptions/SRFileLoggerExceptions.h"
#include "../SRPlatform/Interface/SRLock.h"

namespace SRPlat {

const char* const FileLogger::cShutdownToken = "[[[SD]]]";

FileLogger::FileLogger(const SRString& baseName) : _bShutdown(0), _enqueuedLen(0)
{
  _fileName = baseName.ToString() + SRUtils::PrintUtcDate().ToString() + ".log";
  _fpout = fopen(_fileName.c_str(), "at");
  if (_fpout == nullptr) {
    throw SRCannotOpenLogFileException(_fileName);
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
  std::string sShutdown(cShutdownToken);
  SRLock<SRCriticalSection> csl(_cs);
  if (!_bShutdown) {
    _bShutdown = 1;
    _enqueuedLen += sShutdown.size();
    _qu.push(std::move(sShutdown));
  }
}

void FileLogger::WriterEntry() {
  for (;;)
  {

  }
}

} // namespace SRPlat