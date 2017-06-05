#pragma once
#include "../SRPlatform/Interface/SRException.h"

namespace SRPlat {

class SRPLATFORM_API SRCannotOpenLogFileException : public SRException {
  SRString _fileName;
public:
  explicit SRCannotOpenLogFileException(const std::string& fileName) : _fileName(fileName),
    SRException(SRString::MakeUnowned("Cannot open log file.")) { }
};

} // namespace SRPlat
