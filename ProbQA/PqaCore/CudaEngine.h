// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/BaseCudaEngine.h"
#include "../PqaCore/CudaArray.h"

namespace ProbQA {

template<typename taNumber> class CudaEngine : public BaseCudaEngine {
private: // variables
  //// N questions, K answers, M targets
  // space A: [iQuestion][iAnswer][iTarget] . Guarded by _rws
  CudaArray<taNumber, true> _sA;
  // matrix D: [iQuestion][iTarget] . Guarded by _rws
  CudaArray<taNumber, true> _mD;
  // vector B: [iTarget] . Guarded by _rws
  CudaArray<taNumber, true> _vB;

protected: // methods
  PqaError TrainSpec(const TPqaId nQuestions, const AnsweredQuestion* const pAQs, const TPqaId iTarget,
    const TPqaAmount amount) override final;
  TPqaId ResumeQuizSpec(PqaError& err, const TPqaId nAnswered, const AnsweredQuestion* const pAQs) override final;
  TPqaId NextQuestionSpec(PqaError& err, BaseQuiz *pBaseQuiz) override final;
  TPqaId ListTopTargetsSpec(PqaError& err, BaseQuiz *pBaseQuiz, const TPqaId maxCount,
    RatedTarget *pDest) override final;
  PqaError RecordQuizTargetSpec(BaseQuiz *pBaseQuiz, const TPqaId iTarget, const TPqaAmount amount) override final;
  PqaError AddQsTsSpec(const TPqaId nQuestions, AddQuestionParam *pAqps, const TPqaId nTargets,
    AddTargetParam *pAtps)  override final;
  PqaError CompactSpec(CompactionResult &cr)  override final;

  size_t NumberSize() override final { return sizeof(taNumber); };
  PqaError SaveStatistics(KBFileInfo &kbfi) override final;
  PqaError DestroyQuiz(BaseQuiz *pQuiz) override final;
  PqaError DestroyStatistics() override final;

public: // methods
  explicit CudaEngine(const EngineDefinition& engDef, KBFileInfo *pKbFi);

  TPqaId StartQuiz(PqaError& err) override final;
};

} // namespace ProbQA
