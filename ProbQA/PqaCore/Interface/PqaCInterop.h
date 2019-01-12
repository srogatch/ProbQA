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

typedef struct {
  int64_t _iTarget;
  double _prob; // probability that this target is what the user needs
} CiRatedTarget;

typedef struct {
  int64_t _index;
  double _initAmount;
} CiAddQorTParam; // The parameter for adding a question or target

#pragma pack(pop)

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

PQACORE_API void CiDebugBreak(void);

PQACORE_API uint8_t Logger_Init(void **ppStrErr, const char* baseName);
PQACORE_API void CiReleaseString(void *pvString);

PQACORE_API void* CiGetPqaEngineFactory();
PQACORE_API void* PqaEngineFactory_CreateCpuEngine(void* pvFactory, void **ppError, const CiEngineDefinition *pEngDef);
PQACORE_API void* PqaEngineFactory_LoadCpuEngine(void *pvFactory, void **ppError, const char* filePath,
  uint64_t memPoolMaxBytes);

PQACORE_API void CiReleasePqaError(void *pvErr);
PQACORE_API void* PqaError_ToString(void *pvError, const uint8_t withParams);

PQACORE_API void CiReleasePqaEngine(void *pvEngine);
PQACORE_API void* PqaEngine_Train(void *pvEngine, int64_t nQuestions, const CiAnsweredQuestion* const pAQs,
  const int64_t iTarget, const double amount = 1.0);

PQACORE_API uint8_t PqaEngine_QuestionPermFromComp(void *pvEngine, const int64_t count, int64_t *pIds);
PQACORE_API uint8_t PqaEngine_QuestionCompFromPerm(void *pvEngine, const int64_t count, int64_t *pIds);

PQACORE_API uint8_t PqaEngine_TargetPermFromComp(void *pvEngine, const int64_t count, int64_t *pIds);
PQACORE_API uint8_t PqaEngine_TargetCompFromPerm(void *pvEngine, const int64_t count, int64_t *pIds);

PQACORE_API uint8_t PqaEngine_QuizPermFromComp(void *pvEngine, const int64_t count, int64_t *pIds);
PQACORE_API uint8_t PqaEngine_QuizCompFromPerm(void *pvEngine, const int64_t count, int64_t *pIds);
// Return whether the next permanent has been increased as a result of this operation.
PQACORE_API uint8_t PqaEngine_EnsurePermQuizGreater(void *pvEngine, const int64_t bound);
PQACORE_API uint8_t PqaEngine_RemapQuizPermId(void *pvEngine, const int64_t srcPermId, const int64_t destPermId);

PQACORE_API uint64_t PqaEngine_GetTotalQuestionsAsked(void *pvEngine, void **ppError);
PQACORE_API uint8_t PqaEngine_CopyDims(void *pvEngine, CiEngineDimensions *pDims);
PQACORE_API int64_t PqaEngine_StartQuiz(void *pvEngine, void **ppError);
PQACORE_API int64_t PqaEngine_ResumeQuiz(void *pvEngine, void **ppError, const int64_t nAnswered,
  const CiAnsweredQuestion* const pAQs);
PQACORE_API int64_t PqaEngine_NextQuestion(void *pvEngine, void **ppError, const int64_t iQuiz);
PQACORE_API void* PqaEngine_RecordAnswer(void *pvEngine, const int64_t iQuiz, const int64_t iAnswer);

PQACORE_API int64_t PqaEngine_GetActiveQuestionId(void *pvEngine, void **ppError, const int64_t iQuiz);
PQACORE_API void* PqaEngine_SetActiveQuestion(void *pvEngine, const int64_t iQuiz, const int64_t iQuestion);

PQACORE_API int64_t PqaEngine_ListTopTargets(void *pvEngine, void **ppError, const int64_t iQuiz,
  const int64_t maxCount, CiRatedTarget *pDest);
PQACORE_API void* PqaEngine_RecordQuizTarget(void *pvEngine, const int64_t iQuiz, const int64_t iTarget,
  const double amount = 1.0);
PQACORE_API void* PqaEngine_ReleaseQuiz(void *pvEngine, const int64_t iQuiz);
PQACORE_API void* PqaEngine_SaveKB(void *pvEngine, const char* const filePath, const uint8_t bDoubleBuffer);

//// Second batch of interop implementation
PQACORE_API void* PqaEngine_StartMaintenance(void *pvEngine, const bool forceQuizzes);
PQACORE_API void* PqaEngine_FinishMaintenance(void *pvEngine);
PQACORE_API void* PqaEngine_AddQsTs(void *pvEngine, const int64_t nQuestions, CiAddQorTParam *pAddQuestionParams,
  const int64_t nTargets, CiAddQorTParam *pAddTargetParams);
PQACORE_API void* PqaEngine_RemoveQuestions(void *pvEngine, const int64_t nQuestions, const int64_t *pQIds);
PQACORE_API void* PqaEngine_RemoveTargets(void *pvEngine, const int64_t nTargets, const int64_t *pTIds);
PQACORE_API void* PqaEngine_Compact(void *pvEngine, int64_t *pnQuestions, int64_t const ** const ppOldQuestions,
  int64_t *pnTargets, int64_t const ** const ppOldTargets);
PQACORE_API void CiReleaseCompaction(const int64_t *p);
PQACORE_API void* PqaEngine_Shutdown(void *pvEngine, const char* const saveFilePath = nullptr);
PQACORE_API void* PqaEngine_SetLogger(void *pvEngine, void *pSRLogger);


#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
