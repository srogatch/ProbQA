#pragma once

#include "../PqaCore/Interface/PqaCommon.h"
#include "../PqaCore/Interface/PqaCore.h"

namespace ProbQA {

enum class PqaErrorCode : int64_t {
  None = 0,
  NotImplemented = 1
};

class PQACORE_API IPqaErrorParams {
public:
  // For memory deallocation in the same module where it was allocated.
  void Release();
  virtual ~IPqaErrorParams() { }
};

class PQACORE_API PqaError {
  PqaErrorCode _code;
  IPqaErrorParams *_pParams;

public:
  PqaError() : _code(PqaErrorCode::None), _pParams(nullptr) { }
  //TODO: collect call stack
  PqaError(PqaErrorCode code, IPqaErrorParams *pParams) : _code(code), _pParams(pParams) { }
  PqaError& operator=(const PqaError& fellow) = delete;
  PqaError(const PqaError& fellow) = delete;
  PqaError& operator=(PqaError&& fellow);
  PqaError(PqaError&& fellow);
  ~PqaError();

  bool isOk() const { return _code == PqaErrorCode::None; }
  PqaErrorCode GetCode() const { return _code; }
  IPqaErrorParams* GetParams() const { return _pParams; }
  IPqaErrorParams* DetachParams();
  void Release();
};

} // namespace ProbQA
