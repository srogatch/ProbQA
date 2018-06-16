// Probabilistic Question-Answering system
// @2017 Sarge Rogatch
// This software is distributed under GNU AGPLv3 license. See file LICENSE in repository root for details.

#pragma once

#include "../PqaCore/BaseCpuEngine.fwd.h"
#include "../PqaCore/CEQuiz.fwd.h"
#include "../PqaCore/CEBaseTask.fwd.h"
#include "../PqaCore/CETask.fwd.h"

#include "../PqaCore/BaseEngine.h"

namespace ProbQA {

class BaseCpuEngine : public BaseEngine {
private:
  const SRPlat::SRThreadCount _nLooseWorkers;
  const SRPlat::SRThreadCount _nMemOpThreads;

protected: // variables
  // Most operations are thread-safe already.
  // ChangeStackSize() is not thread-safe, therefore guarded by _maintSwitch intraswitch mode.
  SRPlat::SRThreadPool _tpWorkers;

protected: // methods
  static SRPlat::SRThreadCount CalcMemOpThreads();
  explicit BaseCpuEngine(const EngineDefinition& engDef, const size_t workerStackSize, KBFileInfo *pKbFi);

  PqaError ShutdownWorkers() override final;

public: // Internal interface methods
  SRPlat::SRThreadPool& GetWorkers() { return _tpWorkers; }
  const SRPlat::SRThreadCount GetNLooseWorkers() const { return _nLooseWorkers; }
};

} // namespace ProbQA
