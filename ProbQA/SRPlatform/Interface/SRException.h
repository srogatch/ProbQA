#pragma once

#include "SRPlatform.h"
#include "SRString.h"

namespace SRPlat {

// Can't inherit from std::exception because that's not DLL-exported (causes compiler warning)
class SRPLATFORM_API SRException {
  SRString _message;
public:
  explicit SRException(SRString &&message);
  virtual ~SRException() {}
  virtual SRString GetMsg() const;
  SRString MoveMsg();
};

} // namespace SRPlat
