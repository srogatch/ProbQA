// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCuda/PqaCuda.h"

namespace PqaCuda {
  class PQACUDA_API CudaError {
    int64_t _code;
    SRPlat::SRString _message;

  public:
    explicit CudaError(int64_t code, SRPlat::SRString&& message) : _code(code),
      _message(std::forward<SRPlat::SRString>(message)) { }
  };
}
