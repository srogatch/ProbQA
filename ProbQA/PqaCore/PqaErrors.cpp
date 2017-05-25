#include "stdafx.h"
#include "../PqaCore/Interface/PqaErrors.h"

namespace ProbQA {

void IPqaErrorParams::Release() {
  delete this;
}

PqaError& PqaError::operator=(PqaError&& fellow) {
  if (this != &fellow) {
    Release();
    _code = fellow._code;
    _pParams = fellow.DetachParams();
    fellow._code = PqaErrorCode::None;
  }
  return *this;
}

PqaError::PqaError(PqaError&& fellow) : _code(fellow._code), _pParams(fellow.DetachParams()) {
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
}

} // namespace ProbQA