// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"
#include "../PqaCuda/CudaException.h"
#include "../PqaCuda/Utils.h"

namespace PqaCuda {

CudaException::CudaException(const int64_t cuErr, SRPlat::SRString &&message) : _cuErr(cuErr),
  SRException(message)
{
}

} // namespace PqaCuda
