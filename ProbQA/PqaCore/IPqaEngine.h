#pragma once

// PQA - Probabilistic Question Answering
typedef int64_t TPqaId;
typedef double TPqaAmount;

const TPqaId cInvalidPqaId = -1;

struct AnsweredQuestion {
  TPqaId _iQuestion;
  TPqaId _iAnswer;
};

struct RatedTarget {
  TPqaId _iTarget;
  TPqaAmount _prob; // probability that this target is what the user needs
};

struct CompactionResult {
  //// New counts of targets and questions
  TPqaId _nTargets;
  TPqaId _nQuestions;
  //// i-th item contains the old id for the new id=i
  TPqaId *_pOldTargets; 
  TPqaId *_pOldQuestions;
};

class IPqaEngine {
  // A possibility to train the knowledge base without running a quiz.
  bool Train(const TPqaId nQuestions, const AnsweredQuestion* const pAQs, const TPqaId iTarget,
    const TPqaAmount amount = 1);

  // Returns new quiz ID.
  TPqaId StartQuiz();
  // Start a new quiz with the given answers applied.
  // Returns quiz ID.
  TPqaId ResumeQuiz(const TPqaId nQuestions, const AnsweredQuestion* const pAQs);
  // Returns the ID of the next question to ask.
  // Returns -1 on error (e.g. maintenance in progress).
  TPqaId NextQuestion(const TPqaId iQuiz);
  // Record the user answer for the last question asked. Must be called no more than once for each question.
  bool RecordAnswer(const TPqaId iQuiz, const TPqaId iAnswer);
  // Returns the number of targets written to the destination.
  // Returns -1 on error.
  TPqaId ListTopTargets(const TPqaId iQuiz, const TPqaId maxCount, RatedTarget *pDest);
  // Can be called multiple times for different targets and at different stages in the quiz.
  bool RecordQuizTarget(const TPqaId iQuiz, const TPqaId iTarget, const TPqaAmount amount = 1);
  // Release the resources occupied by the quiz.
  bool ReleaseQuiz(const TPqaId iQuiz);

  //// These maintenance operations are slow when not in maintenance mode, because they have to update all the quizes
  ////   currently in progress.
  TPqaId AddQuestion(const TPqaAmount initialAmount = 1);
  TPqaId AddTarget(const TPqaAmount initialAmount = 1);

  // Save the knowledge base, but not the quizes in progress.
  bool SaveKB(const char* const filePath);
  // Statistics method, especially useful for charging.
  uint64_t GetTotalQuestionsAsked();

  // When |forceQuizes|=false, the function fails if there are any quizes in progress.
  // When |forceQuizes|=true, the function closes all the open quizes.
  // Upon success, the function prohibits starting any new quizes until FinishMaintenance() is called.
  bool StartMaintenance(const bool forceQuizes);
  bool FinishMaintenance();

  //// These methods can only be run in maintenance mode
  bool RemoveQuestion(const TPqaId iQuestion);
  bool RemoveTarget(const TPqaId iTarget);
  bool Compact(CompactionResult &cr);

  //// Helpers to maintenance mode that can be run later
  void ReleaseCompactionResult(CompactionResult &cr);

  //TODO: thread-local error reporting methods
};
