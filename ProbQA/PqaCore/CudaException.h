// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

namespace ProbQA {

class CudaException : public SRPlat::SRException {
  int64_t _cuErr;

public:
  explicit CudaException(const int64_t cuErr, SRPlat::SRString &&message);

  CudaException(const CudaException &fellow) : SRPlat::SRException(fellow), _cuErr(fellow._cuErr) { }
  CudaException(CudaException &&fellow) : SRPlat::SRException(std::forward<SRPlat::SRException>(fellow)),
    _cuErr(fellow._cuErr) { }

  SREXCEPTION_TYPICAL(Cuda);
};

} // namespace ProbQA
