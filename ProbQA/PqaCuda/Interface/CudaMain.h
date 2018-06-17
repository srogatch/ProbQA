// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCuda/Interface/PqaCuda.h"

namespace PqaCuda {

class PQACUDA_API CudaMain {
public:
  static void SetDevice(const int iDevice, const bool bFirstInProcess);
};

} // namespace PqaCuda
