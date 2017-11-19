// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/PqaCore.h"
#include "../PqaCore/Interface/PqaErrorParams.h"
#include "../PqaCore/Interface/PqaErrors.h"

namespace ProbQA {

class PQACORE_API PqaException : public SRPlat::SRException {
  IPqaErrorParams *_pParams;
  PqaErrorCode _errCode;

public:
  explicit PqaException(PqaErrorCode errCode, IPqaErrorParams *pParams, SRPlat::SRString &&message = SRPlat::SRString())
    : SRPlat::SRException(std::forward<SRPlat::SRString>(message)), _pParams(pParams), _errCode(errCode) { }
  virtual ~PqaException() override { delete _pParams; }

  SREXCEPTION_TYPICAL(Pqa);

  PqaErrorCode GetCode() const { return _errCode; }
  IPqaErrorParams* DetachParams() { IPqaErrorParams*  answer = _pParams; _pParams = nullptr; return answer; }
};

} // namespace ProbQA
