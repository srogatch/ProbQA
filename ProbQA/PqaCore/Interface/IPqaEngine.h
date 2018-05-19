// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/PqaErrors.h"
#include "../PqaCore/Interface/PqaCommon.h"
#include "../PqaCore/Interface/PqaCore.h"

namespace ProbQA {

class PQACORE_API IPqaEngine {
public:
  virtual ~IPqaEngine() { }

  // A possibility to train the knowledge base without running a quiz.
  // |pAQs| can contain duplicate questions.
  virtual PqaError Train(const TPqaId nQuestions, const AnsweredQuestion* const pAQs, const TPqaId iTarget,
    const TPqaAmount amount = 1) = 0;

  //// Permanent-compact ID mappers
  virtual bool QuestionPermFromComp(const TPqaId count, TPqaId *ids) = 0;
  virtual bool QuestionCompFromPerm(const TPqaId count, TPqaId *ids) = 0;
  virtual bool TargetPermFromComp(const TPqaId count, TPqaId *ids) = 0;
  virtual bool TargetCompFromPerm(const TPqaId count, TPqaId *ids) = 0;

  // Statistics method, especially useful for charging.
  virtual uint64_t GetTotalQuestionsAsked(PqaError& err) = 0;
  // Get engine dimensions: the number of questions, answers and targets
  virtual const EngineDimensions& GetDims() const = 0;

  //// There must be no concurrent requests on the same quiz. This is not thread-safe.
#pragma region Regular-only mode operations
  // Returns new quiz ID.
  virtual TPqaId StartQuiz(PqaError& err) = 0;
  // Start a new quiz with the given answers applied.
  // Returns quiz ID.
  virtual TPqaId ResumeQuiz(PqaError& err, const TPqaId nAnswered, const AnsweredQuestion* const pAQs) = 0;
  // Returns the ID of the next question to ask. If this method is called without RecordAnswer(), then the active
  //   question is skipped.
  // Returns -1 on error (e.g. when maintenance in progress or when out of questions).
  virtual TPqaId NextQuestion(PqaError& err, const TPqaId iQuiz) = 0;
  // Record the user answer for the last question asked. Must be called no more than once for each question.
  virtual PqaError RecordAnswer(const TPqaId iQuiz, const TPqaId iAnswer) = 0;

  // Returns the number of targets written to the destination.
  // Returns -1 on error.
  virtual TPqaId ListTopTargets(PqaError& err, const TPqaId iQuiz, const TPqaId maxCount, RatedTarget *pDest) = 0;

  // Can be called multiple times for different targets and at different stages in the quiz.
  virtual PqaError RecordQuizTarget(const TPqaId iQuiz, const TPqaId iTarget, const TPqaAmount amount = 1) = 0;
  // Release the resources occupied by the quiz.
  virtual PqaError ReleaseQuiz(const TPqaId iQuiz) = 0;
#pragma endregion

  // Save the knowledge base, but not the quizes in progress.
  // Double buffer uses as much additional memory as the size of the KB, but reduces KB lock duration because the KB
  //   is only locked for the period of copying in memory to the buffer, then saving to disk proceeds without a lock.
  virtual PqaError SaveKB(const char* const filePath, const bool bDoubleBuffer) = 0;

  // When |forceQuizes|=false, the function fails if there are any quizes in progress.
  // When |forceQuizes|=true, the function closes all the open quizes.
  // Upon success, the function prohibits starting any new quizes until FinishMaintenance() is called.
  virtual PqaError StartMaintenance(const bool forceQuizes) = 0;
  virtual PqaError FinishMaintenance() = 0;

#pragma region Maintenance-only mode operations
  //// Initially I thought to allow these in regular mode too, however, if a question is deleted while a quiz that uses
  ////   it is in progress, the resulting training by RecordQuizTarget() would decrease the quality of the KB because
  ////   target probabilities may change without this question answered.
  //// Removal of a target requires recomputation of all target probabilities in each quiz, so this is also undesired
  ////   to keep the engine fast.

  // For the cells where both question and target are added (the intersection), the initial amount for the question
  //   is used and the initial amount for the target is ignored.
  virtual PqaError AddQsTs(const TPqaId nQuestions, AddQuestionParam *pAqps, const TPqaId nTargets,
    AddTargetParam *pAtps) = 0;
  virtual PqaError RemoveQuestions(const TPqaId nQuestions, const TPqaId *pQIds) = 0;
  virtual PqaError RemoveTargets(const TPqaId nTargets, const TPqaId *pTIds) = 0;

  // Compacts questions and targets so that there are no gaps.
  // Fills the CompactionResult structure passed in. A call to ReleaseCompactionResult() is needed to release the
  //   resources after usage of the structure.
  //TODO: make this obligatory before exiting the maintenance mode? So to forbid gaps in regular mode.
  virtual PqaError Compact(CompactionResult &cr) = 0;
#pragma endregion

  //// Control operations
  // Shutdown the engine, optionally saving the data to the given file path. The engine can't be brought back to
  //   operational state after this method is called.
  virtual PqaError Shutdown(const char* const saveFilePath = nullptr) = 0;
  virtual PqaError SetLogger(SRPlat::ISRLogger *pLogger) = 0;
};

} // namespace ProbQA