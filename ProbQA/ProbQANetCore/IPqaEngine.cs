using System;
using System.Collections.Generic;
using System.Text;

namespace ProbQANetCore
{
  public interface IPqaEngine
  {
    bool QuestionPermFromComp(Int64[] ids);
    bool QuestionCompFromPerm(Int64[] ids);
    bool TargetPermFromComp(Int64[] ids);
    bool TargetCompFromPerm(Int64[] ids);

    PqaError Train(Int64 nQuestions, AnsweredQuestion[] AQs, Int64 iTarget, double amount = 1.0);

    UInt64 GetTotalQuestionsAsked(out PqaError err);
    EngineDimensions CopyDims();

    Int64 StartQuiz(out PqaError err);
    Int64 ResumeQuiz(out PqaError err, Int64 nAnswered, AnsweredQuestion[] AQs);
    Int64 NextQuestion(out PqaError err, Int64 iQuiz);
    Int64 GetActiveQuestionId(out PqaError err, Int64 iQuiz);
    PqaError RecordAnswer(Int64 iQuiz, Int64 iAnswer);
    Int64 ListTopTargets(out PqaError err, Int64 iQuiz, RatedTarget[] dest);
    PqaError RecordQuizTarget(Int64 iQuiz, Int64 iTarget, double amount = 1.0);
    PqaError ReleaseQuiz(Int64 iQuiz);

    PqaError SaveKB(string filePath, bool bDoubleBuffer);
  }
}
