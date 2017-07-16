// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/ISRLogger.h"

namespace SRPlat {

class ISRLogCustomizable {
public:
  virtual ~ISRLogCustomizable() { }
  virtual ISRLogger* GetLogger() const = 0;
  virtual void SetLogger(ISRLogger *pLogger) = 0;
};

} // namespace SRPlat
