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

typedef struct {
  int64_t _iQuestion;
  int64_t _iAnswer;
} CiAnsweredQuestion;

typedef struct {
  int64_t _nAnswers;
  int64_t _nQuestions;
  int64_t _nTargets;
} CiEngineDimensions;

#pragma pack(pop)

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

PQACORE_API bool CiLogger_Init(void **ppStrErr, const char* baseName);
PQACORE_API void CiReleaseString(void *pvString);

PQACORE_API void* CiPqaGetEngineFactory();
PQACORE_API void* CiPqaEngineFactory_CreateCpuEngine(void* pvFactory, void **ppError, CiEngineDefinition *pEngDef);
PQACORE_API void* CiqaEngineFactory_LoadCpuEngine(void *pvFactory, void **ppError, const char* filePath,
  uint64_t memPoolMaxBytes);

PQACORE_API void CiReleasePqaError(void *pvErr);
PQACORE_API void* CiPqaError_ToString(void *pvError, const uint8_t withParams);

PQACORE_API void CiReleasePqaEngine(void *pvEngine);
PQACORE_API void* PqaEngine_Train(void *pvEngine, int64_t nQuestions, const CiAnsweredQuestion* const pAQs,
  const int64_t iTarget, const double amount = 1.0);
PQACORE_API uint8_t PqaEngine_QuestionPermFromComp(void *pvEngine, const int64_t count, int64_t *pIds);
PQACORE_API uint8_t PqaEngine_QuestionCompFromPerm(void *pvEngine, const int64_t count, int64_t *pIds);
PQACORE_API uint8_t PqaEngine_TargetPermFromComp(void *pvEngine, const int64_t count, int64_t *pIds);
PQACORE_API uint8_t PqaEngine_TargetCompFromPerm(void *pvEngine, const int64_t count, int64_t *pIds);
PQACORE_API uint64_t PqaEngine_GetTotalQuestionsAsked(void *pvEngine, void **ppError);
PQACORE_API uint8_t PqaEngine_CopyDims(void *pvEngine, CiEngineDimensions *pDims);
PQACORE_API int64_t PqaEngine_StartQuiz(void *pvEngine, void **ppError);
PQACORE_API int64_t PqaEngine_ResumeQuiz(void *pvEngine, void **ppError, const int64_t nAnswered,
  const CiAnsweredQuestion* const pAQs);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
