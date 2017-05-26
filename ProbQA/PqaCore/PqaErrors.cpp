#include "stdafx.h"
#include "../PqaCore/Interface/PqaErrors.h"
#include "../PqaCore/PqaException.h"

using namespace SRPlat;

namespace ProbQA {

void IPqaErrorParams::Release() {
  delete this;
}

PqaError& PqaError::operator=(PqaError&& fellow) {
  if (this != &fellow) {
    Release();
    _code = fellow._code;
    fellow._code = PqaErrorCode::None;
    _pParams = fellow.DetachParams();
    _message = std::move(fellow._message);
  }
  return *this;
}

PqaError::PqaError(PqaError&& fellow) : _code(fellow._code), _pParams(fellow.DetachParams()),
  _message(std::forward<SRPlat::SRString>(fellow._message))
{
  fellow._code = PqaErrorCode::None;
}

PqaError::~PqaError() {
  Release();
}

IPqaErrorParams* PqaError::DetachParams() {
  IPqaErrorParams *pParams = _pParams;
  _pParams = nullptr;
  return pParams;
}

void PqaError::Release() {
  if (_pParams != nullptr) {
    _pParams->Release();
    _pParams = nullptr;
  }
  _code = PqaErrorCode::None;
  _message = SRPlat::SRString();
}

void PqaError::SetFromException(SRPlat::SRException &&ex) {
  PqaException *pPqaEx = dynamic_cast<PqaException*>(&ex);
  if (pPqaEx == nullptr) {
    _code = PqaErrorCode::SRException;
    _message = ex.MoveMsg();
    _pParams = new CommonExceptionErrorParams(ex);
  }
  else {
    _code = pPqaEx->GetCode();
    _message = pPqaEx->MoveMsg();
    _pParams = pPqaEx->DetachParams();
  }
}

void PqaError::SetFromException(const std::exception &ex) {
  _code = PqaErrorCode::StdException;
  _message = SRString::MakeClone(ex.what());
  _pParams = new CommonExceptionErrorParams(ex);
}

} // namespace ProbQA