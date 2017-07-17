// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../SRPlatform/Interface/Exceptions/SRGenericException.h"

namespace SRPlat {

class SRGenericException::Impl {
public: // variables
  std::exception_ptr _ep;

public: // methods
  explicit Impl(const std::exception_ptr &ep) : _ep(ep) { }
};

SRGenericException::SRGenericException(const std::exception_ptr &ep)
  : SRException(GetDefaultMessage()), _pImpl(new Impl(ep))
{
}

SRGenericException::SRGenericException(const SRGenericException &fellow)
  : SRException(fellow), _pImpl(new Impl(*fellow._pImpl)) 
{
}

SRGenericException& SRGenericException::operator=(const SRGenericException &fellow) {
  if (this != &fellow) {
    std::unique_ptr<Impl> pNewImpl(new Impl(*fellow._pImpl));
    _message = fellow._message;
    std::unique_ptr<Impl> pOldImpl(_pImpl);
    _pImpl = pNewImpl.release();
  }
  return *this;
}

SRGenericException::SRGenericException(SRGenericException &&fellow) : SRException(GetDefaultMessage()) {
  std::unique_ptr<Impl> nullImpl(new Impl(nullptr));
  std::swap(_message, fellow._message);
  _pImpl = fellow._pImpl;
  fellow._pImpl = nullImpl.release();
}

SRGenericException& SRGenericException::operator=(SRGenericException &&fellow) {
  if (this != &fellow) {
    std::unique_ptr<Impl> nullImpl(new Impl(nullptr));
    SRString &&nullMessage(GetDefaultMessage());

    _message = std::forward<SRString>(fellow._message);
    fellow._message = std::move(nullMessage);

    std::unique_ptr<Impl> pOldImpl(_pImpl);
    _pImpl = fellow._pImpl;
    fellow._pImpl = nullImpl.release();
  }
  return *this;
}

} // namespace SRPlat
