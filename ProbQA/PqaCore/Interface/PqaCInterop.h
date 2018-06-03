// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/PqaCore.h"

#pragma pack (push, 8)
typedef struct {
  int64_t _nAnswers;
  int64_t _nQuestions;
  int64_t _nTargets;
  uint8_t _precType;
  uint16_t _precExponent;
  uint32_t _precMantissa;
  double _initAmount;
  uint64_t _memPoolMaxBytes;
} CiEngineDefinition;

#pragma pack(pop)

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

PQACORE_API void* CiPqaGetEngineFactory();
PQACORE_API void* CiPqaEngineFactory_CreateCpuEngine(void* pFactory, CiEngineDefinition *pEngDef);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
