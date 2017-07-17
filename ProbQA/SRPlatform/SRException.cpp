// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../SRPlatform/Interface/SRException.h"
#include "../SRPlatform/Interface/SRMessageBuilder.h"

namespace SRPlat {

SRException::SRException(SRString&& message) : _message(std::forward<SRString>(message)) {
}

SRException::SRException(const SRString& message) : _message(message) {
}

const SRString& SRException::GetMsg() const {
  return _message;
}

SRString SRException::MoveMsg() {
  return std::move(_message);
}

} // namespace SRPlat
