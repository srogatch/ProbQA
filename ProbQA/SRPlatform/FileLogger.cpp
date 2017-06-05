#include "stdafx.h"
#include "../SRPlatform/FileLogger.h"
#include "../SRPlatform/Interface/SRUtils.h"
#include "../SRPlatform/Interface/Exceptions/SRFileLoggerExceptions.h"

namespace SRPlat {

FileLogger::FileLogger(const SRString& baseName) {
  std::string fileName = baseName.ToString() + SRUtils::PrintUtcDate().ToString() + ".log";
  _fpout = fopen(fileName.c_str(), "at");
  if (_fpout == nullptr) {
    throw SRCannotOpenLogFileException(fileName);
  }
}

FileLogger::~FileLogger() {

}

bool FileLogger::Log(const Severity s, const SRString& message) {

}

} // namespace SRPlat