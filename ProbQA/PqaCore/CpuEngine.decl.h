// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CpuEngine.fwd.h"
#include "../PqaCore/CEQuiz.fwd.h"
#include "../PqaCore/CECreateQuizOperation.fwd.h"

#include "../PqaCore/BaseCpuEngine.h"
#include "../PqaCore/CENormPriorsTask.h"
#include "../PqaCore/CENormPriorsSubtaskMax.h"
#include "../PqaCore/CENormPriorsSubtaskCorrSum.h"
#include "../PqaCore/CEDivTargPriorsSubtask.h"

namespace ProbQA {

// So long as it's data-only structure, it doesn't need fwd/decl/impl header design.
template<typename taNumber> class CETrainTaskNumSpec;

template<typename taNumber> class CpuEngine : public BaseCpuEngine {
  static_assert(std::is_base_of<SRPlat::SRRealNumber, taNumber>::value, "taNumber must a PqaNumber subclass.");

public: // constants
  static constexpr size_t _cNormPriorsMemReqPerSubtask = std::max({ SRMaxSizeof<CENormPriorsSubtaskMax<taNumber>,
    CENormPriorsSubtaskCorrSum<taNumber>, CEDivTargPriorsSubtask<CENormPriorsTask<taNumber>>>::value,
    SRPlat::SRBucketSummatorPar<taNumber>::_cSubtaskMemReq });

private: // variables
  //// N questions, K answers, M targets

  // space A: [iQuestion][iAnswer][iTarget] . Guarded by _rws
  std::vector<std::vector<SRPlat::SRFastArray<taNumber, false>>> _sA;
  // matrix D: [iQuestion][iTarget] . Guarded by _rws
  std::vector<SRPlat::SRFastArray<taNumber, false>> _mD;
  // vector B: [iTarget] . Guarded by _rws
  SRPlat::SRFastArray<taNumber, false> _vB;

private: // methods

  static size_t CalcWorkerStackSize(const EngineDimensions& dims);

#pragma region Behind StartQuiz() and ResumeQuiz() currently. May be needed by something else.
  TPqaId CreateQuizInternal(CECreateQuizOpBase &op);
#pragma endregion

protected: // Specific methods for this kind of engine
  PqaError TrainSpec(const TPqaId nQuestions, const AnsweredQuestion* const pAQs, const TPqaId iTarget,
    const TPqaAmount amount) override final;
  TPqaId ResumeQuizSpec(PqaError& err, const TPqaId nAnswered, const AnsweredQuestion* const pAQs) override final;
  TPqaId NextQuestionSpec(PqaError& err, BaseQuiz *pBaseQuiz) override final;
  TPqaId ListTopTargetsSpec(PqaError& err, BaseQuiz *pBaseQuiz, const TPqaId maxCount,
    RatedTarget *pDest) override final;
  PqaError RecordQuizTargetSpec(BaseQuiz *pBaseQuiz, const TPqaId iTarget, const TPqaAmount amount) override final;
  PqaError AddQsTsSpec(const TPqaId nQuestions, AddQuestionParam *pAqps, const TPqaId nTargets,
    AddTargetParam *pAtps) override final;
  PqaError CompactSpec(CompactionResult &cr) override final;

  size_t NumberSize() override final;
  PqaError SaveStatistics(KBFileInfo &kbfi) override final;
  PqaError DestroyQuiz(BaseQuiz *pQuiz) override final;
  PqaError DestroyStatistics() override final;
  void UpdateWorkerStacks() override final;

public: // Internal interface methods

  const taNumber& GetA(const TPqaId iQuestion, const TPqaId iAnswer, const TPqaId iTarget) const;
  taNumber& ModA(const TPqaId iQuestion, const TPqaId iAnswer, const TPqaId iTarget);
  
  const taNumber& GetD(const TPqaId iQuestion, const TPqaId iTarget) const;
  taNumber& ModD(const TPqaId iQuestion, const TPqaId iTarget);

  const taNumber& GetB(const TPqaId iTarget) const;
  taNumber& ModB(const TPqaId iTarget);

  PqaError NormalizePriors(CEQuiz<taNumber> &quiz, SRPlat::SRPoolRunner &pr,
    const SRPlat::SRPoolRunner::Split& targSplit);

public: // Client interface methods
  explicit CpuEngine(const EngineDefinition& engDef, KBFileInfo *pKbFi);
  virtual ~CpuEngine() override final;

  virtual TPqaId StartQuiz(PqaError& err) override final;
};

} // namespace ProbQA
