// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "SRPlatform.h"
#include "SRString.h"

namespace SRPlat {

// Can't inherit from std::exception because that's not DLL-exported (causes compiler warning)
class SRPLATFORM_API SRException {
  SRString _message;
public:
  //TODO: collect stack trace
  explicit SRException(SRString &&message);
  virtual ~SRException() {}
  virtual SRString GetMsg() const;
  SRString MoveMsg();
};

} // namespace SRPlat
