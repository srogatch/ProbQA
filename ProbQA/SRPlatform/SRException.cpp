#include "stdafx.h"
#include "../SRPlatform/Interface/SRException.h"

namespace SRPlat {

SRException::SRException(SRString&& message) : _message(std::forward<SRString>(message)) {
}

SRString SRException::GetMsg() const {
  return _message;
}

SRString SRException::MoveMsg() {
  return std::move(_message);
}

} // namespace SRPlat
