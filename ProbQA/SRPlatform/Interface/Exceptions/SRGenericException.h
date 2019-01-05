// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../SRPlatform/Interface/SRException.h"

namespace SRPlat {

class SRPLATFORM_API SRGenericException : public SRException {
  std::exception_ptr _ep;

private: // methods
  static SRString GetDefaultMessage() {
    return SRString::MakeUnowned("Generic exception caugth into std::exception_ptr usually by catch(...) .");
  }

public:
  explicit SRGenericException(const std::exception_ptr &ep);
  SRGenericException(const SRGenericException &fellow);
  SRGenericException& operator=(const SRGenericException &fellow);
  SRGenericException(SRGenericException &&fellow) noexcept;
  SRGenericException& operator=(SRGenericException &&fellow);

  SREXCEPTION_TYPICAL(SRGeneric);
};

} // namespace SRPlat
