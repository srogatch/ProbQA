// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"

using namespace ProbQA;
using namespace SRPlat;
using namespace std;

TEST(Dimensions, CpuIncrease) {
  const TPqaAmount initAm = 1.0;
  PqaError err;
  EngineDefinition ed;
  ed._dims._nAnswers = 5;
  ed._dims._nQuestions = 2;
  ed._dims._nTargets = 2;
  ed._initAmount = initAm;
  ed._prec._type = TPqaPrecisionType::Double;
  IPqaEngine *pEngine = PqaGetEngineFactory().CreateCpuEngine(err, ed);
  ASSERT_TRUE(err.IsOk());
  ASSERT_TRUE(pEngine != nullptr);

  SRFastRandom fr;
  SREntropyAdapter ea(fr);
  vector<AddQuestionParam> aqps;
  AddQuestionParam protoAqp;
  vector<AddTargetParam> atps;
  AddTargetParam protoAtp;
  vector<TPqaAmount> freqs;

  for (TPqaId j = 0; j < 1000; j++) {
    const TPqaId nQuToAdd = ea.Generate(5);
    const TPqaId nTaToAdd = ea.Generate(5);
    
    protoAqp._initialAmount = initAm;
    protoAqp._iQuestion = cInvalidPqaId;
    aqps.assign(nQuToAdd, protoAqp);
    
    protoAtp._initialAmount = initAm;
    protoAtp._iTarget = cInvalidPqaId;
    atps.assign(nTaToAdd, protoAtp);

    const EngineDimensions dimsBefore = pEngine->CopyDims();

    err = pEngine->StartMaintenance(true);
    ASSERT_TRUE(err.IsOk()) << err.ToString(true).ToStd();
    err = pEngine->AddQsTs(nQuToAdd, aqps.data(), nTaToAdd, atps.data());
    ASSERT_TRUE(err.IsOk()) << err.ToString(true).ToStd();
    err = pEngine->FinishMaintenance();
    ASSERT_TRUE(err.IsOk()) << err.ToString(true).ToStd();

    const EngineDimensions dimsAfter = pEngine->CopyDims();
    ASSERT_EQ(dimsBefore._nQuestions + nQuToAdd, dimsAfter._nQuestions);
    ASSERT_EQ(dimsBefore._nTargets + nTaToAdd, dimsAfter._nTargets);

    freqs.resize(dimsAfter._nTargets);
    for (TPqaId iQuestion = 0; iQuestion < dimsAfter._nQuestions; iQuestion++) {
      for (TPqaId iAnswer = 0; iAnswer < dimsAfter._nAnswers; iAnswer++) {
        err = pEngine->CopyATargets(iQuestion, iAnswer, freqs.size(), freqs.data());
        ASSERT_TRUE(err.IsOk()) << err.ToString(true).ToStd();
        for (TPqaId i = 0; i < freqs.size(); i++) {
          ASSERT_NEAR(freqs[i], initAm, initAm / 1e9);
        }
      }
      err = pEngine->CopyDTargets(iQuestion, freqs.size(), freqs.data());
      ASSERT_TRUE(err.IsOk()) << err.ToString(true).ToStd();
      for (TPqaId i = 0; i < freqs.size(); i++) {
        const TPqaAmount expected = dimsAfter._nAnswers * initAm;
        ASSERT_NEAR(freqs[i], expected, expected / 1e9);
      }
    }
    err = pEngine->CopyBTargets(freqs.size(), freqs.data());
    ASSERT_TRUE(err.IsOk()) << err.ToString(true).ToStd();
    for (TPqaId i = 0; i < freqs.size(); i++) {
      ASSERT_NEAR(freqs[i], initAm, initAm / 1e9);
    }

    const TPqaId iQuiz = pEngine->StartQuiz(err);
    ASSERT_TRUE(err.IsOk()) << err.ToString(true).ToStd();
    const TPqaId iNq = pEngine->NextQuestion(err, iQuiz);
    ASSERT_TRUE(err.IsOk()) << err.ToString(true).ToStd();
    // The quiz will be forced in StartMaintenance()
  }
  delete pEngine;
}
