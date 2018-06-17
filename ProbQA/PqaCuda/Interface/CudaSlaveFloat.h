// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCuda/Interface/PqaCuda.h"

namespace PqaCuda {

class CudaSlaveFloatImpl;

class PQACUDA_API CudaSlaveFloat {
  CudaSlaveFloatImpl *_pImpl;
public:
};

} // namespace PqaCuda
