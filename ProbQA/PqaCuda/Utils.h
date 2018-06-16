// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCuda/CudaException.h"

#define CUDA_MUST(statement) do { \
  const cudaError_t status1 = statement; \
  if(status1 == cudaSuccess) break; \
  PqaCuda::CudaException(status1, SRPlat::SRMessageBuilder(SR_FILE_LINE)("CUDA error #")(status1)(": ") \
    (cudaGetErrorString(status1)).GetUnownedSRString()).ThrowMoving(); \
} WHILE_FALSE
