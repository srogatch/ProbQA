// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/CpuEngine.fwd.h"
#include "../PqaCore/CEQuiz.fwd.h"
#include "../PqaCore/CECreateQuizOperation.fwd.h"
#include "../PqaCore/PqaNumber.h"
#include "../PqaCore/BaseCpuEngine.h"

namespace ProbQA {

// So long as it's data-only structure, it doesn't need fwd/decl/impl header design.
template<typename taNumber> class CETrainTaskNumSpec;

template<typename taNumber> class CpuEngine : public BaseCpuEngine {
  static_assert(std::is_base_of<PqaNumber, taNumber>::value, "taNumber must a PqaNumber subclass.");

private: // variables
  // space A: [iAnswer][iQuestion][iTarget] . Guarded by _rws
  std::vector<std::vector<SRPlat::SRFastArray<taNumber, false>>> _sA;
  // matrix D: [iQuestion][iTarget] . Guarded by _rws
  std::vector<SRPlat::SRFastArray<taNumber, false>> _mD;
  // vector B: [iTarget] . Guarded by _rws
  SRPlat::SRFastArray<taNumber, false> _vB;

  std::vector<CEQuiz<taNumber>*> _quizzes; // Guarded by _csQuizReg

private: // methods

#pragma region Behind Train() interface method
  void InitTrainTaskNumSpec(CETrainTaskNumSpec<taNumber>& numSpec, const TPqaAmount amount);
  PqaError TrainInternal(const TPqaId nQuestions, const AnsweredQuestion* const pAQs, const TPqaId iTarget,
    const TPqaAmount amount);
#pragma endregion

#pragma region Behind StartQuiz() and ResumeQuiz() currently. May be needed by something else.
  TPqaId CreateQuizInternal(CECreateQuizOpBase &op);
#pragma endregion

public: // Internal interface methods

  //TODO: move these to implementation file
  const taNumber& GetA(const TPqaId iAnswer, const TPqaId iQuestion, const TPqaId iTarget) const;
  taNumber& ModA(const TPqaId iAnswer, const TPqaId iQuestion, const TPqaId iTarget);
  
  const taNumber& GetD(const TPqaId iQuestion, const TPqaId iTarget) const;
  taNumber& ModD(const TPqaId iQuestion, const TPqaId iTarget);

  const taNumber& GetB(const TPqaId iTarget) const;
  taNumber& ModB(const TPqaId iTarget);

public: // Client interface methods
  explicit CpuEngine(const EngineDefinition& engDef);
  virtual ~CpuEngine() override final;

  virtual PqaError Train(const TPqaId nQuestions, const AnsweredQuestion* const pAQs, const TPqaId iTarget,
    const TPqaAmount amount = 1) override final;

  virtual TPqaId StartQuiz(PqaError& err) override final;
  virtual TPqaId ResumeQuiz(PqaError& err, const TPqaId nQuestions, const AnsweredQuestion* const pAQs) override final;
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
