// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/IPqaEngine.h"
#include "../PqaCore/GapTracker.h"
#include "../PqaCore/MaintenanceSwitch.h"
#include "../PqaCore/Interface/PqaCommon.h"
#include "../PqaCore/KBFileInfo.h"
#include "../PqaCore/PermanentIdManager.h"
#include "../PqaCore/Interface/PqaErrorParams.h"

namespace ProbQA {

class BaseQuiz;

class BaseEngine : public IPqaEngine {
public: // constants
  static constexpr size_t _cMemPoolMaxSimds = size_t(1) << 10;
  static constexpr size_t _cFileBufSize = size_t(1024) * 1024;

public: // types
  typedef SRPlat::SRMemPool<SRPlat::SRSimd::_cLogNBits, _cMemPoolMaxSimds> TMemPool;

protected: // variables
  TMemPool _memPool; // thread-safe itself

  PermanentIdManager _pimQuestions; // Guarded by _rws in maintenance mode. Read-only in regular mode.
  PermanentIdManager _pimTargets; // Guarded by _rws in maintenance mode. Read-only in regular mode.
  PermanentIdManager _pimQuizzes; // Guarded by _csQuizReg . No need to obtain _rws.

  const PrecisionDefinition _precDef;
  EngineDimensions _dims; // Guarded by _rws in maintenance mode. Read-only in regular mode.
  std::atomic<uint64_t> _nQuestionsAsked = 0;

  //// Don't violate the order of obtaining these locks, so to avoid a deadlock.
  //// Actually the locks form directed acyclic graph indicating which locks must be obtained one after another.
  //// However, to simplify the code we list them here topologically sorted.
  mutable MaintenanceSwitch _maintSwitch; // regular/maintenance mode switch
  mutable SRPlat::SRReaderWriterSync _rws; // KB read-write
  SRPlat::SRCriticalSection _csQuizReg; // quiz registry

  std::vector<BaseQuiz*> _quizzes; // Guarded by _csQuizReg
  GapTracker<TPqaId> _quizGaps; // Guarded by _csQuizReg

  GapTracker<TPqaId> _questionGaps; // Guarded by _rws in maintenance mode. Read-only in regular mode.
  GapTracker<TPqaId> _targetGaps; // Guarded by _rws in maintenance mode. Read-only in regular mode.

  //// Cache-insensitive data
  std::atomic<SRPlat::ISRLogger*> _pLogger;

protected: // methods
  explicit BaseEngine(const EngineDefinition& engDef, KBFileInfo *pKbFi);

  TPqaId FindNearestQuestion(const TPqaId iMiddle, const __m256i *pQAsked);

  void AfterStatisticsInit(KBFileInfo *pKbFi);
  bool ReadGaps(GapTracker<TPqaId> &gt, KBFileInfo &kbfi);
  bool WriteGaps(const GapTracker<TPqaId> &gt, KBFileInfo &kbfi);

  PqaError LockedSaveKB(KBFileInfo &kbfi, const bool bDoubleBuffer);
  BaseQuiz* UseQuiz(PqaError& err, const TPqaId iQuiz);

  TPqaId AssignQuiz(BaseQuiz *pQuiz);
  void UnassignQuiz(const TPqaId iQuiz);

  void LockedReleaseQuiz(const TPqaId iQuiz, AggregateErrorParams &aep);

protected: // Specific methods for this engine
  virtual PqaError TrainSpec(const TPqaId nQuestions, const AnsweredQuestion* const pAQs, const TPqaId iTarget,
    const TPqaAmount amount) = 0;
  virtual TPqaId ResumeQuizSpec(PqaError& err, const TPqaId nAnswered, const AnsweredQuestion* const pAQs) = 0;
  virtual TPqaId NextQuestionSpec(PqaError& err, BaseQuiz *pBaseQuiz) = 0;
  virtual TPqaId ListTopTargetsSpec(PqaError& err, BaseQuiz *pBaseQuiz, const TPqaId maxCount,
    RatedTarget *pDest) = 0;
  virtual PqaError RecordQuizTargetSpec(BaseQuiz *pBaseQuiz, const TPqaId iTarget, const TPqaAmount amount) = 0;
  virtual PqaError AddQsTsSpec(const TPqaId nQuestions, AddQuestionParam *pAqps, const TPqaId nTargets,
    AddTargetParam *pAtps) = 0;
  virtual PqaError CompactSpec(CompactionResult &cr) = 0;

  virtual size_t NumberSize() = 0;
  virtual PqaError SaveStatistics(KBFileInfo &kbfi) = 0;
  virtual PqaError DestroyQuiz(BaseQuiz *pQuiz) = 0;
  virtual PqaError DestroyStatistics() = 0;
  virtual PqaError ShutdownWorkers() = 0;
  virtual void UpdateWithDimensions() = 0;

public: // Internal interface methods
  SRPlat::ISRLogger *GetLogger() const { return _pLogger.load(std::memory_order_relaxed); }
  TMemPool& GetMemPool() { return _memPool; }
  SRPlat::SRReaderWriterSync& GetRws() { return _rws; }

  const GapTracker<TPqaId>& GetQuestionGaps() const { return _questionGaps; }
  const GapTracker<TPqaId>& GetTargetGaps() const { return _targetGaps; }

  // Can't be used externally because the dimensions may change when not under a lock
  const EngineDimensions& GetDims() const { return _dims; }

public:
  PqaError Train(const TPqaId nQuestions, const AnsweredQuestion* const pAQs, const TPqaId iTarget,
    const TPqaAmount amount = 1) override final;

  bool QuestionPermFromComp(const TPqaId count, TPqaId *pIds) override final;
  bool QuestionCompFromPerm(const TPqaId count, TPqaId *pIds) override final;
  bool TargetPermFromComp(const TPqaId count, TPqaId *pIds) override final;
  bool TargetCompFromPerm(const TPqaId count, TPqaId *pIds) override final;
  bool QuizPermFromComp(const TPqaId count, TPqaId *pIds) override final;
  bool QuizCompFromPerm(const TPqaId count, TPqaId *pIds) override final;
  bool EnsurePermQuizGreater(const TPqaId bound) override final;
  bool RemapQuizPermId(const TPqaId srcPermId, const TPqaId destPermId) override final;

  uint64_t GetTotalQuestionsAsked(PqaError& err) override final;
  EngineDimensions CopyDims() const override final;

  TPqaId ResumeQuiz(PqaError& err, const TPqaId nAnswered, const AnsweredQuestion* const pAQs) override final;
  TPqaId NextQuestion(PqaError& err, const TPqaId iQuiz) override final;
  PqaError RecordAnswer(const TPqaId iQuiz, const TPqaId iAnswer) override final;
  TPqaId GetActiveQuestionId(PqaError &err, const TPqaId iQuiz) override final;
  PqaError SetActiveQuestion(const TPqaId iQuiz, const TPqaId iQuestion) override final;
  TPqaId ListTopTargets(PqaError& err, const TPqaId iQuiz, const TPqaId maxCount, RatedTarget *pDest) override final;
  PqaError RecordQuizTarget(const TPqaId iQuiz, const TPqaId iTarget, const TPqaAmount amount = 1) override final;
  PqaError ReleaseQuiz(const TPqaId iQuiz) override final;

  virtual PqaError ClearOldQuizzes(const TPqaId maxCount, const double maxAgeSec) override final;

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
};

} // namespace ProbQA
