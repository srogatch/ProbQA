#pragma once

#include "../SRPlatform/Interface/ISRLogger.h"

namespace SRPlat {

class DbgLogger : public ISRLogger {
public:
  virtual bool Log(const Severity sev, const SRString& message) override;
  virtual SRString GetFileName() override;
};

} // namespace SRPlat
