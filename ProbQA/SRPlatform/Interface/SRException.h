#pragma once

#include "SRPlatform.h"
#include "SRString.h"

namespace SRPlat {

// Can't inherit from std::exception because 
class SRPLATFORM_API SRException {
  SRString _message;
public:
  explicit SRException(SRString &&message);
  virtual ~SRException() {}
  virtual SRString GetMsg() const;
  SRString MoveMsg();
};

} // namespace SRPlat
