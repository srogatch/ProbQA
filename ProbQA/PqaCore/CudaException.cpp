// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCore/CudaException.h"

namespace ProbQA {

CudaException::CudaException(const int64_t cuErr, SRPlat::SRString &&message) : _cuErr(cuErr),
  SRException(message)
{
}

} // namespace ProbQA
