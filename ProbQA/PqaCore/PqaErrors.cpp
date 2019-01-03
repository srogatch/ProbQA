// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/Interface/PqaErrors.h"
#include "../PqaCore/PqaException.h"

using namespace SRPlat;

namespace ProbQA {

SRPlat::SRString ToSRString(const PqaErrorCode pec) {
  switch (pec) {
  case PqaErrorCode::None:
    return SRString::MakeUnowned("Success");
  case PqaErrorCode::NotImplemented:
    return SRString::MakeUnowned("Not implemented");
  case PqaErrorCode::SRException:
    return SRString::MakeUnowned("SRException");
  case PqaErrorCode::StdException:
    return SRString::MakeUnowned("std::exception");
  case PqaErrorCode::InsufficientEngineDimensions:
    return SRString::MakeUnowned("Insufficient engine dimensions");
  case PqaErrorCode::MaintenanceModeChangeInProgress:
    return SRString::MakeUnowned("Maintenance mode change is in progress");
  case PqaErrorCode::MaintenanceModeAlreadyThis:
    return SRString::MakeUnowned("Maintenance mode is already this");
  case PqaErrorCode::ObjectShutDown:
    return SRString::MakeUnowned("Object is shut(ting) down");
  case PqaErrorCode::IndexOutOfRange:
    return SRString::MakeUnowned("Index is out of range");
  case PqaErrorCode::Aggregate:
    return SRString::MakeUnowned("Aggregate error");
  case PqaErrorCode::NegativeCount:
    return SRString::MakeUnowned("The count is negative");
  case PqaErrorCode::NonPositiveAmount:
    return SRString::MakeUnowned("The amount is not positive");
  case PqaErrorCode::AbsentId:
    return SRString::MakeUnowned("The ID is absent from KB");
  case PqaErrorCode::WrongMode:
    return SRString::MakeUnowned("An attempt to execute an operation in a wrong mode");
  case PqaErrorCode::UnhandledCase:
    return SRString::MakeUnowned("Unhandled case");
  case PqaErrorCode::I64Underflow:
    return SRString::MakeUnowned("Underflow of a 64-bit integer");
  case PqaErrorCode::QuestionsExhausted:
    return SRString::MakeUnowned("Engine has run out of questions");
  case PqaErrorCode::NoQuizActiveQuestion:
    return SRString::MakeUnowned("No active question in the quiz");
  case PqaErrorCode::CantOpenFile:
    return SRString::MakeUnowned("Cannot open file");
  case PqaErrorCode::FileOp:
    return SRString::MakeUnowned("File operation failed");
  case PqaErrorCode::QuizzesActive:
    return SRString::MakeUnowned("There are still active quizzes");
  case PqaErrorCode::NullArgument:
    return SRString::MakeUnowned("Expected non-null argument");
  default: {
    std::string message("Unhandled");
    message += std::to_string(static_cast<int64_t>(pec));
    return SRString::MakeClone(message.c_str(), message.size());
  } }
}

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

PqaError& PqaError::SetFromException(SRPlat::SRException &&ex) {
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
  return *this;
}

PqaError& PqaError::SetFromException(const std::exception &ex) {
  _code = PqaErrorCode::StdException;
  _message = SRString::MakeClone(ex.what());
  _pParams = new CommonExceptionErrorParams(ex);
  return *this;
}

SRString PqaError::ToString(const bool withParams) {
  SRMessageBuilder mb;
  mb.AppendChar('[')(ToSRString(_code))("] message=[")(_message);
  if (!withParams) {
    return mb.AppendChar(']').GetOwnedSRString();
  }
  mb("] [");
  if (_pParams == nullptr) {
    mb("nullptr");
  }
  else {
    mb(_pParams->ToString());
  }
  mb.AppendChar(']');
  return mb.GetOwnedSRString();
}

} // namespace ProbQA