// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRPlatform.h"
#include "../SRPlatform/Interface/SRString.h"

namespace SRPlat {

#define SREXCEPTION_TYPICAL(nameVar) \
  virtual nameVar##Exception* Clone() override { return new nameVar##Exception(*this); } \
  virtual nameVar##Exception* Move() override { return new nameVar##Exception(std::move(*this)); } \
  virtual void ThrowMoving() override { throw std::move(*this); } \
  virtual void ThrowCopying() override { throw *this; }

#define DERIVE_SREXCEPTION(derivedName, baseName) \
class SRPLATFORM_API derivedName##Exception : public baseName##Exception { \
public: \
  explicit derivedName##Exception(SRString &&message) : baseName##Exception(std::forward<SRString>(message)) { } \
  derivedName##Exception(const derivedName##Exception &fellow) \
    : baseName##Exception(fellow) { } \
  derivedName##Exception(derivedName##Exception &&fellow) noexcept \
    : baseName##Exception(std::forward<baseName##Exception>(fellow)) { } \
  SREXCEPTION_TYPICAL(derivedName); \
};

// Can't inherit from std::exception because that's not DLL-exported (causes compiler warning)
class SRPLATFORM_API SRException {
protected:
  SRString _message;
public:
  //TODO: collect stack trace
  explicit SRException(const SRString& message);
  explicit SRException(SRString &&message);
  virtual ~SRException() {}
  SRException(const SRException &fellow) : _message(fellow._message) { }
  SRException(SRException &&fellow) noexcept : _message(std::forward<SRString>(fellow._message)) { }

  virtual SRException* Clone() { return new SRException(*this); }
  virtual SRException* Move() { return new SRException(std::move(*this)); }
  virtual void ThrowMoving() { throw std::move(*this); }
  virtual void ThrowCopying() { throw *this; }

  const SRString& GetMsg() const;
  SRString MoveMsg();
  virtual SRString ToString() const { return _message; }
};

} // namespace SRPlat
