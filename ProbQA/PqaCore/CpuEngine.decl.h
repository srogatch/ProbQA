// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CpuEngine.fwd.h"
#include "../PqaCore/CEQuiz.fwd.h"
#include "../PqaCore/CECreateQuizOperation.fwd.h"

#include "../PqaCore/KBFileInfo.h"
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

  std::vector<CEQuiz<taNumber>*> _quizzes; // Guarded by _csQuizReg

private: // methods

  static size_t CalcWorkerStackSize(const EngineDefinition& engDef);

#pragma region Behind Train() interface method
  PqaError TrainInternal(const TPqaId nQuestions, const AnsweredQuestion* const pAQs, const TPqaId iTarget,
    const TPqaAmount amount);
#pragma endregion

#pragma region Behind StartQuiz() and ResumeQuiz() currently. May be needed by something else.
  TPqaId CreateQuizInternal(CECreateQuizOpBase &op);
#pragma endregion

  CEQuiz<taNumber>* UseQuiz(PqaError& err, const TPqaId iQuiz);

  PqaError LockedSaveKB(SRPlat::SRSmartFile &sf, const bool bDoubleBuffer, const char* const filePath);

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

  virtual PqaError Train(const TPqaId nQuestions, const AnsweredQuestion* const pAQs, const TPqaId iTarget,
    const TPqaAmount amount = 1) override final;

  virtual TPqaId StartQuiz(PqaError& err) override final;
  virtual TPqaId ResumeQuiz(PqaError& err, const TPqaId nAnswered, const AnsweredQuestion* const pAQs) override final;
  virtual TPqaId NextQuestion(PqaError& err, const TPqaId iQuiz) override final;
  virtual PqaError RecordAnswer(const TPqaId iQuiz, const TPqaId iAnswer) override final;
  virtual TPqaId ListTopTargets(PqaError& err, const TPqaId iQuiz, const TPqaId maxCount, RatedTarget *pDest)
    override final;
  virtual PqaError RecordQuizTarget(const TPqaId iQuiz, const TPqaId iTarget, const TPqaAmount amount = 1)
    override final;
  virtual PqaError ReleaseQuiz(const TPqaId iQuiz) override final;


  virtual PqaError SaveKB(const char* const filePath, const bool bDoubleBuffer) override final;
  virtual uint64_t GetTotalQuestionsAsked(PqaError& err) override final;

  virtual PqaError StartMaintenance(const bool forceQuizes) override final;
  virtual PqaError FinishMaintenance() override final;

  virtual PqaError AddQuestions(TPqaId nQuestions, AddQuestionParam *pAqps) override final;
  virtual PqaError AddTargets(TPqaId nTargets, AddTargetParam *pAtps) override final;
  virtual PqaError RemoveQuestions(const TPqaId nQuestions, const TPqaId *pQIds) override final;
  virtual PqaError RemoveTargets(const TPqaId nTargets, const TPqaId *pTIds) override final;


  virtual PqaError Compact(CompactionResult &cr) override final;

  virtual PqaError ReleaseCompactionResult(CompactionResult &cr) override final;

  virtual PqaError Shutdown(const char* const saveFilePath = nullptr) override final;
  virtual PqaError SetLogger(SRPlat::ISRLogger *pLogger) override final;
};

} // namespace ProbQA
