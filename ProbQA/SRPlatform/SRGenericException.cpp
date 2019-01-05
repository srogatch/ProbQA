// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../SRPlatform/Interface/Exceptions/SRGenericException.h"

namespace SRPlat {

SRGenericException::SRGenericException(const std::exception_ptr &ep)
  : SRException(GetDefaultMessage()), _ep(ep)
{
}

SRGenericException::SRGenericException(const SRGenericException &fellow)
  : SRException(fellow), _ep(fellow._ep)
{
}

SRGenericException& SRGenericException::operator=(const SRGenericException &fellow) {
  if (this != &fellow) {
    std::exception_ptr ep = fellow._ep;
    _message = fellow._message;
    _ep = std::move(ep);
  }
  return *this;
}

SRGenericException::SRGenericException(SRGenericException &&fellow) noexcept : SRException(GetDefaultMessage()) {
  std::swap(_message, fellow._message);
  _ep = std::forward<std::exception_ptr>(fellow._ep);
}

SRGenericException& SRGenericException::operator=(SRGenericException &&fellow) {
  if (this != &fellow) {
    SRString &&nullMessage(GetDefaultMessage());
    _ep = std::forward<std::exception_ptr>(fellow._ep);
    _message = std::forward<SRString>(fellow._message);
    fellow._message = std::move(nullMessage);
  }
  return *this;
}

} // namespace SRPlat
