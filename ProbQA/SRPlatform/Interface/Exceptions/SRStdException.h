// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once
#include "../SRPlatform/Interface/SRException.h"
#include "../SRPlatform/Interface/SRMessageBuilder.h"

namespace SRPlat {

class SRPLATFORM_API SRStdException : public SRException {
  SRString _typeName;
  SRString _stdexMsg;
public:
  explicit SRStdException(const std::exception& ex)
    : _typeName(SRString::MakeUnowned(typeid(ex).name())), _stdexMsg(SRString::MakeOwned(ex.what())),
    SRException(SRString::MakeUnowned("Converted from a (descendant of) std::exception.")) { }

  SRStdException(const SRStdException &fellow) : _typeName(fellow._typeName), _stdexMsg(fellow._stdexMsg),
    SRException(fellow) { }

  SRStdException(SRStdException &&fellow) : _typeName(std::forward<SRString>(fellow._typeName)),
    _stdexMsg(std::forward<SRString>(fellow._stdexMsg)), SRException(std::forward<SRException>(fellow)) { }

  SREXCEPTION_TYPICAL(SRStd);

  virtual SRString ToString() override final {
    return SRMessageBuilder()(GetMsg())(" Type name [")(_typeName)("]. std::exception message [")(_stdexMsg)("].")
      .GetOwnedSRString();
  }
};

} // namespace SRPlat
