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
  ed._initAmount = 0.1;
  ed._prec._type = TPqaPrecisionType::Double;
  IPqaEngine *pEngine = PqaGetEngineFactory().CreateCpuEngine(err, ed);
  ASSERT_TRUE(err.IsOk());
  ASSERT_TRUE(pEngine != nullptr);

  SRFastRandom fr;
  SREntropyAdapter ea(fr);

  constexpr int64_t cnTrials = 10 * 1000;
  constexpr int64_t cMaxQuizLen = 100;
  constexpr int64_t cnTopRated = 10;

  int64_t nCorrect = 0;
  int64_t nTrials = 0;
  for (int64_t i = 1; ; i++) {
    const uint64_t totQAsked = pEngine->GetTotalQuestionsAsked(err);
    if (totQAsked > 3 * 1000 * 1000) {
      if (nTrials >= cnTrials) {
        break;
      }
      nTrials++;
    }
    ASSERT_TRUE(err.IsOk());
    const TPqaId guess = ea.Generate<TPqaId>(ed._dims._nTargets);
    const TPqaId iQuiz = pEngine->StartQuiz(err);
    ASSERT_TRUE(err.IsOk());
    ASSERT_TRUE(iQuiz != cInvalidPqaId);
    int64_t j = 0;
    for (; j < cMaxQuizLen; j++) {
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
        if (nTrials > 0) {
          nCorrect++;
        }
        putchar('+');
        break;
      }
    }
    if (j >= cMaxQuizLen) {
      putchar('-');
    }
    err = pEngine->RecordQuizTarget(iQuiz, guess);
    ASSERT_TRUE(err.IsOk());
    err = pEngine->ReleaseQuiz(iQuiz);
    ASSERT_TRUE(err.IsOk());
  }
  ASSERT_GE(nCorrect, nTrials * 0.98);
}
