#pragma once

#include "../PqaCore/Interface/IPqaEngine.h"
#include "../PqaCore/GapTracker.h"
#include "../PqaCore/MaintenanceSwitch.h"
#include "../PqaCore/Interface/PqaCommon.h"
#include "../PqaCore/PqaNumber.h"
#include "../PqaCore/CESubtask.h" // needed for Kind enum

namespace ProbQA {

template<typename taNumber> class CEQuiz;
template<typename taNumber> class CETask;
template<typename taNumber> class CETrainSubtaskDistrib;
template<typename taNumber> class CETrainSubtaskAdd;
template<typename taNumber> class CETrainTaskNumSpec;
template<typename taNumber, bool taCache> class CECalcTargetPriorsSubtask;
template<typename taNumber> class CECalcTargetPriorsTaskNumSpec;
class CECreateQuizResume;

template<typename taNumber = PqaNumber> class CpuEngine : public IPqaEngine {
  static_assert(std::is_base_of<PqaNumber, taNumber>::value, "taNumber must a PqaNumber subclass.");

public: // constants
  static const TPqaId cMinAnswers = 2;
  static const TPqaId cMinQuestions = 1;
  static const TPqaId cMinTargets = 2;
  static const uint8_t cLogSimdBits = 8; // AVX2, 256 bits
  static const size_t cSimdBytes = 1 << (cLogSimdBits - 3);
  static const size_t cMemPoolMaxSimds = 1 << 10;
  static const uint8_t cNumsPerSimd;

public: // types
  typedef SRPlat::SRSpinSync<32> TStpSync;
  typedef SRPlat::SRMemPool<cLogSimdBits, cMemPoolMaxSimds> TMemPool;

private: // variables
  //TODO: refactor these vectors to a custom vector data structure that inits and copies with AVX2 skipping cache
  // space A: [iAnswer][iQuestion][iTarget] . Guarded by _rws
  std::vector<std::vector<std::vector<taNumber, SRPlat::SRAlignedAllocator<taNumber, cSimdBytes>>>> _sA;
  // matrix D: [iQuestion][iTarget] . Guarded by _rws
  std::vector<std::vector<taNumber, SRPlat::SRAlignedAllocator<taNumber, cSimdBytes>>> _mD;
  // vector B: [iTarget] . Guarded by _rws
  std::vector<taNumber, SRPlat::SRAlignedAllocator<taNumber, cSimdBytes>> _vB;
  // aggregate C: sum of B[iTarget] . Guarded by _rws
  taNumber _aC;
  GapTracker<TPqaId> _questionGaps; // Guarded by _rws
  GapTracker<TPqaId> _targetGaps; // Guarded by _rws
  EngineDimensions _dims; // Guarded by _maintSwitch
  uint64_t _nQuestionsAsked = 0; // Guarded by _rws
  
  //// Don't violate the order of obtaining these locks, so to avoid a deadlock.
  //// Actually the locks form directed acyclic graph indicating which locks must be obtained one after another.
  //// However, to simplify the code we list them here topologically sorted.
  MaintenanceSwitch _maintSwitch;
  SRPlat::SRReaderWriterSync _rws; // KB read-write
  SRPlat::SRCriticalSection _csQuizReg; // quiz registry
  SRPlat::SRCriticalSection _csWorkers; // queue for async workers
  TStpSync _stpSync; // SubTask Pool Sync
  
  uint8_t _shutdownRequested : 1; // guarded by _csWorkers
  std::queue<CESubtask<taNumber>*> _quWork; // guarded by _csWorkers
  SRPlat::SRConditionVariable _haveWork;

  GapTracker<TPqaId> _quizGaps; // Guarded by _csQuizReg
  std::vector<CEQuiz<taNumber>*> _quizzes; // Guarded by _csQuizReg

  std::vector<std::vector<CESubtask<taNumber>*>> _stPool; // Guarded by _stpSync
  TMemPool _memPool; // thread-safe itself

  //// Cache-insensitive data
  // The size of this vector must not change after construction of CpuEngine, because it's accessed without locks.
  std::vector<std::thread> _workers;
  std::atomic<SRPlat::ISRLogger*> _pLogger;
private: // methods
  void WorkerEntry();
  void RunSubtask(CESubtask<taNumber> &ceSt);

  CESubtask<taNumber>* CreateSubtask(const typename CESubtask<taNumber>::Kind kind);
  void DeleteSubtask(CESubtask<taNumber> *pSubtask);

  // Returns the number of vectors for the given number of numbers.
  static TPqaId GetSimdCount(const TPqaId nNumbers);

#pragma region Behind Train() interface method
  void RunTrainDistrib(CETrainSubtaskDistrib<taNumber> &tsd);
  void RunTrainAdd(CETrainSubtaskAdd<taNumber> &tsa);
  void InitTrainTaskNumSpec(CETrainTaskNumSpec<taNumber>& numSpec, const TPqaAmount amount);
  // Update target totals |_vB| and grand total |_aC|
  void TrainUpdateTargetTotals(const TPqaId iTarget, const CETrainTaskNumSpec<taNumber>& numSpec);
  PqaError TrainInternal(const TPqaId nQuestions, const AnsweredQuestion* const pAQs, const TPqaId iTarget,
    const TPqaAmount amount);
#pragma endregion

#pragma region Behind StartQuiz() and ResumeQuiz() currently. May be needed by something else.
  // Assumes _rws is locked at least in shared mode. Assumes pDest reserves integer number of SIMDs.
  template<bool taCache> PqaError CalcTargetPriors(taNumber *pDest);
  template<bool taCache> void RunCalcTargetPriors(CECalcTargetPriorsSubtask<taNumber, taCache>& ctps);
  void InitCalcTargetPriorsNumSpec(CECalcTargetPriorsTaskNumSpec<taNumber> &numSpec);
  template<typename taOperation> TPqaId CreateQuizInternal(taOperation &op);
#pragma endregion

public: // Internal interface methods
  SRPlat::ISRLogger *GetLogger() { return _pLogger.load(std::memory_order_relaxed); }
  const EngineDimensions& GetDims() { return _dims; }
  TMemPool& GetMemPool() { return _memPool; }

  template<typename taSubtask> taSubtask* AcquireSubtask();
  void ReleaseSubtask(CESubtask<taNumber> *pSubtask);

  void WakeWorkersWait(CETask<taNumber> &task);

  void UpdatePriorsWithAnsweredQuestions(CECreateQuizResume& resumeOp);

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
