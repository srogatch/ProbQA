// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#include "stdafx.h"

using namespace ProbQA;
using namespace SRPlat;

TEST(Dimensions, CpuIncrease) {
  PqaError err;
  EngineDefinition ed;
  ed._dims._nAnswers = 5;
  ed._dims._nQuestions = 2;
  ed._dims._nTargets = 2;
  ed._initAmount = 1;
  ed._prec._type = TPqaPrecisionType::Double;
  IPqaEngine *pEngine = PqaGetEngineFactory().CreateCpuEngine(err, ed);
  ASSERT_TRUE(err.IsOk());
  ASSERT_TRUE(pEngine != nullptr);

  SRFastRandom fr;
  SREntropyAdapter ea(fr);
  std::vector<AddQuestionParam> aqps;
  AddQuestionParam protoAqp;
  std::vector<AddTargetParam> atps;
  AddTargetParam protoAtp;
  for (TPqaId i = 0; i < 1000; i++) {
    const TPqaId nQuToAdd = ea.Generate(5);
    const TPqaId nTaToAdd = ea.Generate(5);
    
    protoAqp._initialAmount = 1;
    protoAqp._iQuestion = cInvalidPqaId;
    aqps.assign(nQuToAdd, protoAqp);
    
    protoAtp._initialAmount = 1;
    protoAtp._iTarget = cInvalidPqaId;
    atps.assign(nTaToAdd, protoAtp);

    err = pEngine->AddQsTs(nQuToAdd, aqps.data(), nTaToAdd, atps.data());
    ASSERT_TRUE(err.IsOk());

    EngineDimensions dims = pEngine->CopyDims();
  }
}
