// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/BaseCudaEngine.h"

namespace ProbQA {

template<typename taNumber> class CudaEngine : public BaseCudaEngine {
private: // variables
  //// N questions, K answers, M targets
  // space A: [iQuestion][iAnswer][iTarget] . Guarded by _rws
  CudaArray<taNumber> _sA;
  // matrix D: [iQuestion][iTarget] . Guarded by _rws
  CudaArray<taNumber> _mD;
  // vector B: [iTarget] . Guarded by _rws
  CudaArray<taNumber> _vB;

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
  void UpdateWithDimensions() override final;

public: // Internal interface methods
  taNumber* GetMD() { return _mD.Get(); }
  taNumber* GetSA() { return _sA.Get(); }

public: // methods
  explicit CudaEngine(const EngineDefinition& engDef, KBFileInfo *pKbFi);

  TPqaId StartQuiz(PqaError& err) override final;
};

} // namespace ProbQA
