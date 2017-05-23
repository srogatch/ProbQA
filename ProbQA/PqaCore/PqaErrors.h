#pragma once

#include "PqaCommon.h"
#include "PqaCore.h"

namespace ProbQA {

enum class PqaErrorCode : int64_t {
  None = 0,
  NotImplemented = 1
};

class PQACORE_API IPqaErrorParams {
public:
  virtual ~IPqaErrorParams() { }
};

class PQACORE_API PqaError {
  PqaErrorCode _code;
  IPqaErrorParams *_pParams;

public:
  bool isOk() const { return _code == PqaErrorCode::None; }
  PqaErrorCode GetCode() const;
  IPqaErrorParams* GetParams() const;
  // This method must be called for each error returned. For convenient minimum error handling, there is an option to
  //   just log the error before releasing the error resources.
  void Release(const bool bLog = true);
};

} // namespace ProbQA
