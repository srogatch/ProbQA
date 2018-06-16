// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/IPqaEngine.h"

namespace ProbQA {

class BaseCudaEngine : public IPqaEngine {
public:
  ~BaseCudaEngine() override;

  PqaError Train(const TPqaId nQuestions, const AnsweredQuestion* const pAQs, const TPqaId iTarget,
    const TPqaAmount amount = 1) override final;

  bool QuestionPermFromComp(const TPqaId count, TPqaId *pIds) override final;
  bool QuestionCompFromPerm(const TPqaId count, TPqaId *pIds) override final;
  bool TargetPermFromComp(const TPqaId count, TPqaId *pIds) override final;
  bool TargetCompFromPerm(const TPqaId count, TPqaId *pIds) override final;

  uint64_t GetTotalQuestionsAsked(PqaError& err) override final;
  EngineDimensions CopyDims() const override final;

  TPqaId StartQuiz(PqaError& err) override final;
  TPqaId ResumeQuiz(PqaError& err, const TPqaId nAnswered, const AnsweredQuestion* const pAQs) override final;
  TPqaId NextQuestion(PqaError& err, const TPqaId iQuiz) override final;
  PqaError RecordAnswer(const TPqaId iQuiz, const TPqaId iAnswer) override final;
  TPqaId GetActiveQuestionId(PqaError &err, const TPqaId iQuiz) override final;
  TPqaId ListTopTargets(PqaError& err, const TPqaId iQuiz, const TPqaId maxCount, RatedTarget *pDest) override final;
  PqaError RecordQuizTarget(const TPqaId iQuiz, const TPqaId iTarget, const TPqaAmount amount = 1) override final;
  PqaError ReleaseQuiz(const TPqaId iQuiz) override final;

  PqaError SaveKB(const char* const filePath, const bool bDoubleBuffer) override final;

  virtual PqaError StartMaintenance(const bool forceQuizes) override final;
  virtual PqaError FinishMaintenance() override final;

  PqaError AddQsTs(const TPqaId nQuestions, AddQuestionParam *pAqps, const TPqaId nTargets,
    AddTargetParam *pAtps) override final;
  PqaError RemoveQuestions(const TPqaId nQuestions, const TPqaId *pQIds) override final;
  PqaError RemoveTargets(const TPqaId nTargets, const TPqaId *pTIds) override final;

  PqaError Compact(CompactionResult &cr) override final;

  PqaError Shutdown(const char* const saveFilePath = nullptr) override final;
  PqaError SetLogger(SRPlat::ISRLogger *pLogger) override final;

protected:
  explicit BaseCudaEngine(const EngineDefinition& engDef);
};

} // namespace ProbQA
