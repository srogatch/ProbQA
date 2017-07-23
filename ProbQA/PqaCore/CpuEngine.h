// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/Interface/IPqaEngine.h"
#include "../PqaCore/GapTracker.h"
#include "../PqaCore/MaintenanceSwitch.h"
#include "../PqaCore/Interface/PqaCommon.h"
#include "../PqaCore/PqaNumber.h"

namespace ProbQA {

template<typename taNumber> class CEQuiz;
template<typename taNumber> class CETask;
template<typename taNumber> class CETrainSubtaskDistrib;
template<typename taNumber> class CETrainSubtaskAdd;
template<typename taNumber> class CETrainTaskNumSpec;
template<typename taNumber> class CECreateQuizResume;

template<typename taNumber = PqaNumber> class CpuEngine : public IPqaEngine {
  static_assert(std::is_base_of<PqaNumber, taNumber>::value, "taNumber must a PqaNumber subclass.");

public: // constants
  static const TPqaId cMinAnswers = 2;
  static const TPqaId cMinQuestions = 1;
  static const TPqaId cMinTargets = 2;
  static const size_t cMemPoolMaxSimds = 1 << 10;

public: // types
  typedef SRPlat::SRSpinSync<32> TStpSync;
  typedef SRPlat::SRMemPool<SRPlat::SRSimd::_cLogNBits, cMemPoolMaxSimds> TMemPool;

private: // variables
  TMemPool _memPool; // thread-safe itself
  SRPlat::SRThreadPool _tpWorkers; // thread-safe itself

  // space A: [iAnswer][iQuestion][iTarget] . Guarded by _rws
  std::vector<std::vector<SRPlat::SRFastArray<taNumber, false>>> _sA;
  // matrix D: [iQuestion][iTarget] . Guarded by _rws
  std::vector<SRPlat::SRFastArray<taNumber, false>> _mD;
  // vector B: [iTarget] . Guarded by _rws
  SRPlat::SRFastArray<taNumber, false> _vB;

  EngineDimensions _dims; // Guarded by _rws in maintenance mode. Read-only in regular mode.
  const SRPlat::SRThreadPool::TThreadCount _nMemOpThreads;
  uint64_t _nQuestionsAsked = 0; // Guarded by _rws

  //// Don't violate the order of obtaining these locks, so to avoid a deadlock.
  //// Actually the locks form directed acyclic graph indicating which locks must be obtained one after another.
  //// However, to simplify the code we list them here topologically sorted.
  MaintenanceSwitch _maintSwitch; // regular/maintenance mode switch
  SRPlat::SRReaderWriterSync _rws; // KB read-write
  SRPlat::SRCriticalSection _csQuizReg; // quiz registry
  
  //TODO: remove after refactoring to SRThreadPool
  //SRPlat::SRCriticalSection _csWorkers; // queue for async workers
  //TODO: remove after refactoring to SRThreadPool
  //TStpSync _stpSync; // SubTask Pool Sync

  GapTracker<TPqaId> _quizGaps; // Guarded by _csQuizReg
  std::vector<CEQuiz<taNumber>*> _quizzes; // Guarded by _csQuizReg

  GapTracker<TPqaId> _questionGaps; // Guarded by _rws in maintenance mode. Read-only in regular mode.
  GapTracker<TPqaId> _targetGaps; // Guarded by _rws in maintenance mode. Read-only in regular mode.
  
  //TODO: remove after refactoring to SRThreadPool
  //uint8_t _shutdownRequested : 1; // guarded by _csWorkers
  //TODO: remove after refactoring to SRThreadPool
  //std::queue<CESubtask<taNumber>*> _quWork; // guarded by _csWorkers
  //TODO: remove after refactoring to SRThreadPool
  //SRPlat::SRConditionVariable _haveWork;

  //TODO: remove after refactoring to SRThreadPool
  //std::vector<std::vector<CESubtask<taNumber>*>> _stPool; // Guarded by _stpSync

  //// Cache-insensitive data
  //TODO: remove after refactoring to SRThreadPool
  // The size of this vector must not change after construction of CpuEngine, because it's accessed without locks.
  //std::vector<std::thread> _workers;
  std::atomic<SRPlat::ISRLogger*> _pLogger;

private: // methods
  static SRPlat::SRThreadPool::TThreadCount CalcMemOpThreads();
  static SRPlat::SRThreadPool::TThreadCount CalcCompThreads();

  template<typename taSubtask, typename taCallback> PqaError SplitAndRunSubtasks(const size_t nWorkers,
    CETask<taNumber> &task, const size_t nItems, void *pSubtaskMem, const taCallback &onVisit);

#pragma region Behind Train() interface method
  void RunTrainDistrib(CETrainSubtaskDistrib<taNumber> &tsd);
  void RunTrainAdd(CETrainSubtaskAdd<taNumber> &tsa);
  void InitTrainTaskNumSpec(CETrainTaskNumSpec<taNumber>& numSpec, const TPqaAmount amount);
  PqaError TrainInternal(const TPqaId nQuestions, const AnsweredQuestion* const pAQs, const TPqaId iTarget,
    const TPqaAmount amount);
#pragma endregion

#pragma region Behind StartQuiz() and ResumeQuiz() currently. May be needed by something else.
  template<typename taOperation> TPqaId CreateQuizInternal(taOperation &op);
#pragma endregion

public: // Internal interface methods
  SRPlat::ISRLogger *GetLogger() { return _pLogger.load(std::memory_order_relaxed); }
  const EngineDimensions& GetDims() { return _dims; }
  TMemPool& GetMemPool() { return _memPool; }

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
