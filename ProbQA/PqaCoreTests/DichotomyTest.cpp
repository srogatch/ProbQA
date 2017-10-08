// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"

using namespace ProbQA;
using namespace SRPlat;

TEST(DichotomyTest, Main) {
  PqaError err;
  EngineDefinition ed;
  ed._dims._nAnswers = 5;
  ed._dims._nQuestions = 1000;
  ed._dims._nTargets = 1000;
  ed._prec._type = TPqaPrecisionType::Double;
  IPqaEngine *pEngine = PqaGetEngineFactory().CreateCpuEngine(err, ed);
  ASSERT_TRUE(err.IsOk());
  ASSERT_TRUE(pEngine != nullptr);

  SRFastRandom fr;
  SREntropyAdapter ea(fr);

  constexpr int64_t cnTrainings = 1000 * 1000;
  constexpr int64_t cnTrials = 3333;
  constexpr int64_t cMaxQuizLen = 100;
  constexpr int64_t cMaxTrialLen = 30;
  constexpr int64_t cnTopRated = 10;

  for (int64_t i = 0; i < cnTrainings + cnTrials; i++) {
    if ((i & 255) == 0) {
      printf("*");
    }
    const TPqaId guess = ea.Generate<TPqaId>(ed._dims._nTargets);
    const TPqaId iQuiz = pEngine->StartQuiz(err);
    ASSERT_TRUE(err.IsOk());
    ASSERT_TRUE(iQuiz != cInvalidPqaId);
    for (int64_t j = 0; j < cMaxQuizLen; j++) {
      const TPqaId iQuestion = pEngine->NextQuestion(err, iQuiz);
      ASSERT_TRUE(err.IsOk());
      ASSERT_TRUE(iQuestion != cInvalidPqaId);
      TPqaId iAnswer;
      if (guess < iQuestion - 32) {
        iAnswer = 0;
      }
      else if (iQuestion - 32 <= guess && guess < iQuestion) {
        iAnswer = 1;
      }
      else if (iQuestion == guess) {
        iAnswer = 2;
      }
      else if (iQuestion < guess && guess <= iQuestion + 32) {
        iAnswer = 3;
      }
      else if(guess > iQuestion + 32) {
        iAnswer = 4;
      }
      else {
        FAIL();
      }
      err = pEngine->RecordAnswer(iQuiz, iAnswer);
      ASSERT_TRUE(err.IsOk());

      RatedTarget rts[cnTopRated];
      const TPqaId nListed = pEngine->ListTopTargets(err, iQuiz, cnTopRated, rts);
      ASSERT_TRUE(err.IsOk());
      ASSERT_EQ(nListed, cnTopRated);

      TPqaId posInTop = cInvalidPqaId;
      for (TPqaId k = 0; k < cnTopRated; k++) {
        if (rts[k]._iTarget == guess) {
          posInTop = k;
          break;
        }
      }
      if (posInTop != cInvalidPqaId) {
        printf("[guess=%" PRId64 ",top=%" PRId64 ",after=%" PRId64 "]", int64_t(guess), int64_t(posInTop), int64_t(j));
        break;
      }
      if (i >= cnTrainings && j >= cMaxTrialLen) {
        for (TPqaId k = 0; k < cnTopRated; k++) {
          printf(" [%g; %" PRId64 "] ", rts[k]._prob, rts[k]._iTarget);
        }
        printf("\n");
        FAIL() << "guess=" << guess;
      }
    }
    err = pEngine->RecordQuizTarget(iQuiz, guess);
    ASSERT_TRUE(err.IsOk());
    pEngine->ReleaseQuiz(iQuiz);
  }
}
