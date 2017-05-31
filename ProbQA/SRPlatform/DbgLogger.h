#pragma once

#include "../SRPlatform/Interface/ISRLogger.h"

namespace SRPlat {

class DbgLogger : public ISRLogger {
public:
  virtual bool Log(const Severity s, const SRString& message) override;
};

} // namespace SRPlat
