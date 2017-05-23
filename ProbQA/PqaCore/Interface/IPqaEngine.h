#pragma once

#include "PqaErrors.h"
#include "PqaCommon.h"
#include "PqaCore.h"

namespace ProbQA {

class PQACORE_API IPqaEngine {
  // A possibility to train the knowledge base without running a quiz.
  virtual PqaError Train(const TPqaId nQuestions, const AnsweredQuestion* const pAQs, const TPqaId iTarget,
    const TPqaAmount amount = 1) = 0;

  // Returns new quiz ID.
  virtual TPqaId StartQuiz(PqaError& err) = 0;
  // Start a new quiz with the given answers applied.
  // Returns quiz ID.
  virtual TPqaId ResumeQuiz(PqaError& err, const TPqaId nQuestions, const AnsweredQuestion* const pAQs) = 0;
  // Returns the ID of the next question to ask.
  // Returns -1 on error (e.g. maintenance in progress).
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

  // Save the knowledge base, but not the quizes in progress.
  // Double buffer uses as much additional memory as the size of the KB, but reduces KB lock duration because the KB
  //   is only locked for the period of copying in memory to the buffer, then saving to disk proceeds without a lock.
  virtual PqaError SaveKB(const char* const filePath, const bool bDoubleBuffer) = 0;
  // Statistics method, especially useful for charging.
  virtual uint64_t GetTotalQuestionsAsked(PqaError& err) = 0;

  // When |forceQuizes|=false, the function fails if there are any quizes in progress.
  // When |forceQuizes|=true, the function closes all the open quizes.
  // Upon success, the function prohibits starting any new quizes until FinishMaintenance() is called.
  virtual PqaError StartMaintenance(const bool forceQuizes) = 0;
  virtual PqaError FinishMaintenance() = 0;

  //// These maintenance operations are slow when not in maintenance mode, because they have to update all the quizes
  ////   currently in progress.
  virtual TPqaId AddQuestion(PqaError& err, const TPqaAmount initialAmount = 1) = 0;
  virtual PqaError AddQuestions(TPqaId nQuestions, AddQuestionParam *pAqps) = 0;

  virtual TPqaId AddTarget(PqaError& err, const TPqaAmount initialAmount = 1) = 0;
  virtual PqaError AddTargets(TPqaId nTargets, AddTargetParam *pAtps) = 0;

  virtual PqaError RemoveQuestion(const TPqaId iQuestion) = 0;
  virtual PqaError RemoveQuestions(const TPqaId nQuestions, const TPqaId *pQIds) = 0;

  virtual PqaError RemoveTarget(const TPqaId iTarget) = 0;
  virtual PqaError RemoveTargets(const TPqaId nTargets, const TPqaId *pTIds) = 0;

  //// These methods can only be run in maintenance mode
  // Compacts questions and targets so that there are no gaps.
  // Fills the CompactionResult structure passed in. A call to ReleaseCompactionResult() is needed to release the
  //   resources after usage of the structure.
  virtual PqaError Compact(CompactionResult &cr) = 0;

  //// Helpers to maintenance mode that can be run later
  virtual PqaError ReleaseCompactionResult(CompactionResult &cr) = 0;
};

} // namespace ProbQA