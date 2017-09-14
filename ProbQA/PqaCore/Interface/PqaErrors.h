// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/PqaCommon.h"
#include "../PqaCore/Interface/PqaCore.h"

namespace ProbQA {

enum class PqaErrorCode : int64_t {
  None = 0, // Success
  NotImplemented = 1, // NotImplementedErrorParams
  SRException = 2, // CommonExceptionErrorParams
  StdException = 3, // CommonExceptionErrorParams
  InsufficientEngineDimensions = 4,
  MaintenanceModeChangeInProgress = 5, // MaintenanceModeErrorParams
  MaintenanceModeAlreadyThis = 6, // MaintenanceModeErrorParams
  // The object is shut(ting) down
  ObjectShutDown = 7, // ObjectShutDownErrorParams
  IndexOutOfRange = 8, // IndexOutOfRangeErrorParams
  Internal = 9, // InternalErrorParams
  // Aggregate error consisting of multiple errors
  Aggregate = 10, // AggregateErrorParams
  NegativeCount = 11, // NegativeCountErrorParams
  NonPositiveAmount = 12, // NonPositiveAmountErrorParams
  AbsentId = 13, // AbsentIdErrorParams
  WrongMode = 14, // No error params
  UnhandledCase = 15, // No error params
  I64Underflow = 16 // I64UnderflowErrorParams
};

SRPlat::SRString ToSRString(const PqaErrorCode pec);

class PQACORE_API IPqaErrorParams {
public:
  IPqaErrorParams() { }
  IPqaErrorParams(const IPqaErrorParams&) = delete;
  IPqaErrorParams& operator=(const IPqaErrorParams&) = delete;
  // For memory deallocation in the same module where it was allocated.
  void Release();
  virtual ~IPqaErrorParams() { }
  virtual SRPlat::SRString ToString() = 0;
};

class PQACORE_API PqaError {
  PqaErrorCode _code;
  IPqaErrorParams *_pParams;
  SRPlat::SRString _message;

public:
  PqaError() : _code(PqaErrorCode::None), _pParams(nullptr) { }
  //TODO: collect call stack
  PqaError(PqaErrorCode code, IPqaErrorParams *pParams, SRPlat::SRString&& message = SRPlat::SRString())
    : _code(code), _pParams(pParams), _message(message) { }
  PqaError& operator=(const PqaError& fellow) = delete;
  PqaError(const PqaError& fellow) = delete;
  PqaError& operator=(PqaError&& fellow);
  PqaError(PqaError&& fellow);
  ~PqaError();

  bool IsOk() const { return _code == PqaErrorCode::None; }
  PqaErrorCode GetCode() const { return _code; }
  IPqaErrorParams* GetParams() const { return _pParams; }
  const SRPlat::SRString& GetMessage() const { return _message; }
  IPqaErrorParams* DetachParams();
  void Release();

  // Also handles PqaException
  PqaError& SetFromException(SRPlat::SRException &&ex);
  PqaError& SetFromException(const std::exception &ex);

  SRPlat::SRString ToString(const bool withParams);
};

} // namespace ProbQA
