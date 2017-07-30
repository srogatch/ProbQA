// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/PqaNumber.h"
#include "../PqaCore/BaseCpuEngine.h"

namespace ProbQA {

template<typename taNumber> class CEQuiz;
template<typename taNumber> class CETrainTaskNumSpec;
template<typename taNumber> class CECreateQuizResume;

template<typename taNumber = PqaNumber> class CpuEngine : public BaseCpuEngine {
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
  template<typename taOperation> TPqaId CreateQuizInternal(taOperation &op);
#pragma endregion

public: // Internal interface methods

  const taNumber& GetA(const TPqaId iAnswer, const TPqaId iQuestion, const TPqaId iTarget) const {
    return _sA[SRPlat::SRCast::ToSizeT(iAnswer)][SRPlat::SRCast::ToSizeT(iQuestion)][SRPlat::SRCast::ToSizeT(iTarget)];
  }
  taNumber& ModA(const TPqaId iAnswer, const TPqaId iQuestion, const TPqaId iTarget) {
    return _sA[SRPlat::SRCast::ToSizeT(iAnswer)][SRPlat::SRCast::ToSizeT(iQuestion)][SRPlat::SRCast::ToSizeT(iTarget)];
  }
  
  const taNumber& GetD(const TPqaId iQuestion, const TPqaId iTarget) const {
    return _mD[SRPlat::SRCast::ToSizeT(iQuestion)][SRPlat::SRCast::ToSizeT(iTarget)];
  }
  taNumber& ModD(const TPqaId iQuestion, const TPqaId iTarget) {
    return _mD[SRPlat::SRCast::ToSizeT(iQuestion)][SRPlat::SRCast::ToSizeT(iTarget)];
  }

  const taNumber& GetB(const TPqaId iTarget) const {
    return _vB[SRPlat::SRCast::ToSizeT(iTarget)];
  }
  taNumber& ModB(const TPqaId iTarget) {
    return _vB[SRPlat::SRCast::ToSizeT(iTarget)];
  }

  void UpdatePriorsWithAnsweredQuestions(CECreateQuizResume<taNumber>& resumeOp);

public: // Client interface methods
  explicit CpuEngine(const EngineDefinition& engDef);
  virtual ~CpuEngine() override;

  virtual PqaError Train(const TPqaId nQuestions, const AnsweredQuestion* const pAQs, const TPqaId iTarget,
    const TPqaAmount amount = 1) override;

  virtual TPqaId StartQuiz(PqaError& err) override;
  virtual TPqaId ResumeQuiz(PqaError& err, const TPqaId nQuestions, const AnsweredQuestion* const pAQs) override;
  virtual TPqaId NextQuestion(PqaError& err, const TPqaId iQuiz) override;
  virtual PqaError RecordAnswer(const TPqaId iQuiz, const TPqaId iAnswer) override;
  virtual TPqaId ListTopTargets(PqaError& err, const TPqaId iQuiz, const TPqaId maxCount, RatedTarget *pDest) override;
  virtual PqaError RecordQuizTarget(const TPqaId iQuiz, const TPqaId iTarget, const TPqaAmount amount = 1) override;
  virtual PqaError ReleaseQuiz(const TPqaId iQuiz) override;


  virtual PqaError SaveKB(const char* const filePath, const bool bDoubleBuffer) override;
  virtual uint64_t GetTotalQuestionsAsked(PqaError& err) override;

  virtual PqaError StartMaintenance(const bool forceQuizes) override;
  virtual PqaError FinishMaintenance() override;

  virtual PqaError AddQuestions(TPqaId nQuestions, AddQuestionParam *pAqps) override;
  virtual PqaError AddTargets(TPqaId nTargets, AddTargetParam *pAtps) override;
  virtual PqaError RemoveQuestions(const TPqaId nQuestions, const TPqaId *pQIds) override;
  virtual PqaError RemoveTargets(const TPqaId nTargets, const TPqaId *pTIds) override;


  virtual PqaError Compact(CompactionResult &cr) override;

  virtual PqaError ReleaseCompactionResult(CompactionResult &cr) override;

  virtual PqaError Shutdown(const char* const saveFilePath = nullptr) override;
  virtual PqaError SetLogger(SRPlat::ISRLogger *pLogger) override;
};

} // namespace ProbQA
