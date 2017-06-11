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
  ObjectShutDown = 7, // ObjectShutDownErrorParams . The object is shut(ting) down
  IndexTooLarge = 8 // IndexTooLargeErrorParams
};

SRPlat::SRString ToSRString(const PqaErrorCode pec);

class PQACORE_API IPqaErrorParams {
public:
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
  PqaError(PqaErrorCode code, IPqaErrorParams *pParams, SRPlat::SRString message = SRPlat::SRString())
    : _code(code), _pParams(pParams), _message(message) { }
  PqaError& operator=(const PqaError& fellow) = delete;
  PqaError(const PqaError& fellow) = delete;
  PqaError& operator=(PqaError&& fellow);
  PqaError(PqaError&& fellow);
  ~PqaError();

  bool isOk() const { return _code == PqaErrorCode::None; }
  PqaErrorCode GetCode() const { return _code; }
  IPqaErrorParams* GetParams() const { return _pParams; }
  const SRPlat::SRString& GetMessage() const { return _message; }
  IPqaErrorParams* DetachParams();
  void Release();

  void SetFromException(SRPlat::SRException &&ex);
  void SetFromException(const std::exception &ex);

  SRPlat::SRString ToString(const bool withParams);
};

} // namespace ProbQA
