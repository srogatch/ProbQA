#pragma once

#include "../PqaCore/Interface/IPqaEngine.h"
#include "../PqaCore/GapTracker.h"
#include "../PqaCore/MaintenanceSwitch.h"
#include "../PqaCore/Interface/PqaCommon.h"
#include "../PqaCore/PqaNumber.h"

namespace ProbQA {

template<typename taNumber = PqaNumber> class CpuEngine : public IPqaEngine {
  static_assert(std::is_base_of<PqaNumber, taNumber>::value, "taNumber must a PqaNumber subclass.");

public: // constants
  static const TPqaId cMinAnswers = 2;
  static const TPqaId cMinQuestions = 1;
  static const TPqaId cMinTargets = 2;
  static const size_t cDataAlign = sizeof(__m256);

private: // variables
  std::vector<std::vector<std::vector<taNumber, SRPlat::SRAlignedAllocator<taNumber, cDataAlign>>>> _cA; // cube A
  std::vector<std::vector<taNumber, SRPlat::SRAlignedAllocator<taNumber, cDataAlign>>> _mD; // matrix D
  std::vector<taNumber, SRPlat::SRAlignedAllocator<taNumber, cDataAlign>> _vB; // vector B
  GapTracker<TPqaId> _questionGaps;
  GapTracker<TPqaId> _targetGaps;
  EngineDimensions _dims;
  taNumber _initAmount;
  uint64_t _nQuestionsAsked = 0;
  
  //// Don't violate the order of obtaining these locks, so to avoid a deadlock.
  MaintenanceSwitch _maintSwitch; // first-entry lock
  SRPlat::SRReaderWriterSync _rws; // second-entry lock

  std::vector<std::thread> _workers;
  uint64_t _shutdownRequested : 1;

  std::atomic<SRPlat::ISRLogger*> _pLogger;

private: // methods
  void WorkerEntry();

public:
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

  virtual TPqaId AddQuestion(PqaError& err, const TPqaAmount initialAmount = 1) override;
  virtual PqaError AddQuestions(TPqaId nQuestions, AddQuestionParam *pAqps) override;

  virtual TPqaId AddTarget(PqaError& err, const TPqaAmount initialAmount = 1) override;
  virtual PqaError AddTargets(TPqaId nTargets, AddTargetParam *pAtps) override;

  virtual PqaError RemoveQuestion(const TPqaId iQuestion) override;
  virtual PqaError RemoveQuestions(const TPqaId nQuestions, const TPqaId *pQIds) override;

  virtual PqaError RemoveTarget(const TPqaId iTarget) override;
  virtual PqaError RemoveTargets(const TPqaId nTargets, const TPqaId *pTIds) override;


  virtual PqaError Compact(CompactionResult &cr) override;

  virtual PqaError ReleaseCompactionResult(CompactionResult &cr) override;

  virtual PqaError Shutdown(const char* const saveFilePath = nullptr) override;
  virtual PqaError SetLogger(SRPlat::ISRLogger *pLogger) override;
};

} // namespace ProbQA
